//! Mixed decode serving for a partially covered warm-table route.
//!
//! Issue #373: a decode token whose routed experts are partly warm and partly
//! cold gathers the covered assignments from the retained warm table and the
//! missing assignments from a streamed page of exactly the missing experts.
//! Restoring the original assignment order before one weighted reduction keeps
//! the arithmetic identical to the single-page path, so mixed serving cannot
//! change generated tokens through floating-point reassociation.

use astronomical_runtime_integration::{MlxArray, MlxRuntime};

use super::paged_execution::RoutedExpertAssignmentOutputs;
use crate::expert_paging::ExpertPageRoutePartition;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::RoutedExpertCoverage;
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5ExpertPager;
use crate::qwen3_5_moe::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_moe::model::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::qwen3_5_moe::model::cached_plus_streamed_page_route::Qwen3_5MoECachedPlusStreamedPageRoute;
use crate::qwen3_5_moe::model::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use crate::qwen3_5_moe::model::routing::qwen3_5_moe_unsorted_expert_weighted_sum;
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

/// Reshapes one partial route side to the batched assignment layout the gather
/// consumes. The route builder emits compact one-dimensional arrays; the
/// single-page path feeds three-dimensional `[batch, token, assignment]`
/// arrays, so mixed serving must match that contract exactly.
fn route_side_arrays(
    runtime: &MlxRuntime,
    page_slot_indices: &MlxArray,
    scores: &MlxArray,
) -> Result<(MlxArray, MlxArray), Qwen3_5ExecutionError> {
    let slot_shape = page_slot_indices.shape();
    let score_shape = scores.shape();
    if slot_shape.len() != 1 || score_shape.len() != 1 || slot_shape != score_shape {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a partial route side must pair one-dimensional slots and scores",
        });
    }
    let assignment_count =
        *slot_shape
            .first()
            .ok_or_else(|| Qwen3_5ExecutionError::InvalidInput {
                description: "a partial route side must not be empty",
            })?;
    let batched_shape = [1, 1, assignment_count];
    Ok((
        runtime.reshape(page_slot_indices, &batched_shape)?,
        runtime.reshape(scores, &batched_shape)?,
    ))
}

impl Qwen3_5Model {
    /// Runs one decode expert forward partly from retained RAM and partly from
    /// storage (issue #373).
    ///
    /// The retained side gathers the covered assignments from the warm table
    /// and the missing side gathers the streamed page, so each partial output
    /// holds a disjoint subset of the rows the single-page path would have
    /// produced. Restoring the original assignment order before one weighted
    /// reduction keeps the arithmetic identical to the single-page path, so
    /// mixed serving cannot change generated tokens through floating-point
    /// reassociation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_cached_plus_streamed_route_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        retained_expert_weights: &Qwen3_5PagedExpertWeights,
        cached_plus_streamed_route: &Qwen3_5MoECachedPlusStreamedPageRoute,
        missing_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_scores: &MlxArray,
        concatenated_assignment_order: &[usize],
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let sparse_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |performance_attribution| {
                let (retained_slot_indices, _retained_scores) = route_side_arrays(
                    &self.runtime,
                    &cached_plus_streamed_route.retained_page_slot_indices,
                    &cached_plus_streamed_route.retained_scores,
                )?;
                let (missing_slot_indices, _missing_scores) = route_side_arrays(
                    &self.runtime,
                    &cached_plus_streamed_route.missing_page_slot_indices,
                    &cached_plus_streamed_route.missing_scores,
                )?;
                let retained_rows = match self.streamed_expert_assignment_outputs(
                    hidden_states,
                    retained_expert_weights,
                    &retained_slot_indices,
                    performance_attribution,
                )? {
                    RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs } => {
                        assignment_outputs
                    }
                    RoutedExpertAssignmentOutputs::Sorted { .. } => {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description: "mixed decode routing must stay unsorted",
                        });
                    }
                };
                let missing_rows = match self.streamed_expert_assignment_outputs(
                    hidden_states,
                    missing_expert_weights,
                    &missing_slot_indices,
                    performance_attribution,
                )? {
                    RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs } => {
                        assignment_outputs
                    }
                    RoutedExpertAssignmentOutputs::Sorted { .. } => {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description: "mixed decode routing must stay unsorted",
                        });
                    }
                };
                let combined_output = qwen3_5_moe_combine_partial_route_outputs_for_tests(
                    &self.runtime,
                    &retained_rows,
                    &missing_rows,
                    concatenated_assignment_order,
                )?;
                // One weighted reduction over the restored original assignment
                // order consumes the same values in the same order as the
                // single-page path, so mixed serving cannot change generated
                // tokens through floating-point reassociation.
                let sparse_expert_output = qwen3_5_moe_unsorted_expert_weighted_sum(
                    &self.runtime,
                    &combined_output,
                    selected_scores,
                )?;
                self.combine_paged_sparse_and_shared_outputs(
                    hidden_states,
                    mixture_of_experts_weights,
                    &sparse_expert_output,
                    should_use_compiled_elementwise_graphs,
                )
            },
        )?;
        performance_attribution.record_counter(PerformanceCounter::HotExpertMixedRouteCount, 1);
        Ok(sparse_output)
    }

    /// Serves one decode token whose routed experts are partly warm and partly
    /// cold.
    ///
    /// Issue #373: the retained page covers only some of the token's routed
    /// experts, so this streams exactly the missing experts, builds the compact
    /// cached-plus-streamed route, and runs one mixed forward. The warm-insert
    /// offer for the streamed experts stays inside the streaming call, so a
    /// mixed forward still warms what it read.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_mixed_decode_route_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        route_order_expert_ids: &[usize],
        route_coverage: &RoutedExpertCoverage,
        expert_capacity: usize,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let retained_expert_ids = &route_coverage.retained_expert_ids;
        let missing_expert_ids = &route_coverage.missing_expert_ids;
        let (retained_expert_weights, retained_page_manifest) = self
            .cached_packed_page(layer_index, retained_expert_ids, expert_capacity)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a partially covered route lost its retained warm experts",
            })?;
        let (missing_expert_weights, missing_page_manifest) = self
            .stream_operation_local_routed_experts(
                expert_pager,
                layer_index,
                1,
                missing_expert_ids,
                paged_prefill_execution_mode
                    == Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
        let route_partition =
            retained_page_manifest.partition_route_assignments(route_order_expert_ids);
        let cached_plus_streamed_route = Qwen3_5MoECachedPlusStreamedPageRoute::build(
            &self.runtime,
            selected_indices,
            selected_scores,
            &route_partition,
            &retained_page_manifest,
            &missing_page_manifest,
        )
        .map_err(|_route_error| Qwen3_5ExecutionError::InvalidInput {
            description: "the cached-plus-streamed route could not be built",
        })?;
        let concatenated_assignment_order =
            concatenated_assignment_order(&route_partition, route_order_expert_ids.len());
        self.forward_moe_cached_plus_streamed_route_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &retained_expert_weights,
            &cached_plus_streamed_route,
            &missing_expert_weights,
            selected_scores,
            &concatenated_assignment_order,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}

fn row_count_before_last(shape: &[i32]) -> Result<i32, Qwen3_5ExecutionError> {
    if shape.len() < 2 {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "a partial route output must include an assignment axis",
        });
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i32, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| Qwen3_5ExecutionError::InvalidInput {
                    description: "a partial route assignment count overflowed",
                })
        })
}

/// Maps each original assignment position to its row in the retained-then-
/// missing concatenation, so one gather restores the single-page row order.
fn concatenated_assignment_order(
    route_partition: &ExpertPageRoutePartition,
    assignment_count: usize,
) -> Vec<usize> {
    let mut concatenated_position_by_assignment = vec![0_usize; assignment_count];
    for (retained_row, assignment_position) in route_partition
        .retained_assignment_positions
        .iter()
        .enumerate()
    {
        concatenated_position_by_assignment[*assignment_position] = retained_row;
    }
    for (missing_row, assignment_position) in route_partition
        .missing_assignment_positions
        .iter()
        .enumerate()
    {
        concatenated_position_by_assignment[*assignment_position] =
            route_partition.retained_assignment_positions.len() + missing_row;
    }
    concatenated_position_by_assignment
}

#[doc(hidden)]
pub fn qwen3_5_moe_combine_partial_route_outputs_for_tests(
    runtime: &MlxRuntime,
    retained_output: &MlxArray,
    missing_output: &MlxArray,
    concatenated_assignment_order: &[usize],
) -> Result<MlxArray, Qwen3_5ExecutionError> {
    let retained_shape = retained_output.shape();
    let missing_shape = missing_output.shape();
    let feature_dimension =
        *retained_shape
            .last()
            .ok_or_else(|| Qwen3_5ExecutionError::InvalidInput {
                description: "a partial route output must not be scalar",
            })?;
    if missing_shape.last() != Some(&feature_dimension) {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "partial route outputs must share one feature dimension",
        });
    }
    let retained_assignment_count = row_count_before_last(&retained_shape)?;
    let missing_assignment_count = row_count_before_last(&missing_shape)?;
    let retained_rows = runtime.reshape(
        retained_output,
        &[retained_assignment_count, feature_dimension],
    )?;
    let missing_rows = runtime.reshape(
        missing_output,
        &[missing_assignment_count, feature_dimension],
    )?;
    let concatenated_rows = runtime.concatenate_axis(&[&retained_rows, &missing_rows], 0)?;
    let total_assignment_count = retained_assignment_count + missing_assignment_count;
    if concatenated_assignment_order.len() != total_assignment_count as usize {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "the partial-route assignment order must cover every assignment",
        });
    }
    let assignment_order = concatenated_assignment_order
        .iter()
        .map(|concatenated_position| {
            u32::try_from(*concatenated_position).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "a partial-route assignment position exceeds the u32 range",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let assignment_order_array =
        runtime.array_from_u32(&assignment_order, &[total_assignment_count])?;
    let ordered_rows = runtime.take_axis(&concatenated_rows, &assignment_order_array, 0)?;
    let restored_rows = runtime.reshape(
        &ordered_rows,
        &[
            retained_shape.first().copied().unwrap_or(1),
            retained_shape.get(1).copied().unwrap_or(1),
            total_assignment_count,
            feature_dimension,
        ],
    );
    restored_rows.map_err(|_runtime_error| Qwen3_5ExecutionError::InvalidInput {
        description: "the restored partial-route rows could not be reshaped",
    })
}
