//! Transfers execution-required expert reads into retained ownership without rereading storage.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_paging::expert_pager::{ExpertPagingError, Qwen3_5ExpertPager};
use crate::{
    ExpertLayerResidencyTarget, PerformanceAttribution, PerformanceCounter, PerformanceOperation,
    RetainedExpertLayerCommitOutcome,
};

use super::Qwen3_5MoEPagedPrefillExecutionMode;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

impl Qwen3_5Model {
    // Paging dependencies stay explicit instead of adding a request-context abstraction.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_layer_store_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        route_token_count: i32,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        should_execute_token_projections_separately: bool,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        known_sorted_unique_expert_ids: Option<&[usize]>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let copied_sorted_unique_expert_ids;
        let sorted_unique_expert_ids = match known_sorted_unique_expert_ids {
            Some(sorted_unique_expert_ids) => sorted_unique_expert_ids,
            None => {
                copied_sorted_unique_expert_ids =
                    self.copy_sorted_unique_expert_ids(selected_indices)?;
                &copied_sorted_unique_expert_ids
            }
        };
        let streamed_expert_ids = if route_token_count > 1 {
            (0..expert_pager.layer_plan(layer_index)?.expert_capacity).collect::<Vec<_>>()
        } else {
            sorted_unique_expert_ids.to_vec()
        };
        let (streamed_expert_weights, page_manifest) = expert_pager
            .load_rust_streamed_expert_layer(
                &self.runtime,
                layer_index,
                &streamed_expert_ids,
                performance_attribution,
            )?;
        if let Some(retained_expert_layers) = self.retained_expert_layers.as_ref() {
            retained_expert_layers.borrow_mut().record_disk_load(
                streamed_expert_ids.len(),
                page_manifest.source_manifests.len(),
            );
        }
        performance_attribution.record_counter(
            if route_token_count > 1 {
                PerformanceCounter::MandatoryPrefillExpertSourcePayloadBytes
            } else {
                PerformanceCounter::MandatoryDecodeExpertSourcePayloadBytes
            },
            page_manifest.payload_byte_count,
        );
        performance_attribution.measure_operation(
            PerformanceOperation::ExpertPagingDiagnosticLogging,
            |_performance_attribution| {
                tracing::debug!(
                    layer_index,
                    route_expert_count = sorted_unique_expert_ids.len(),
                    streamed_expert_count = streamed_expert_ids.len(),
                    streamed_payload_bytes = page_manifest.payload_byte_count,
                    ?paged_prefill_execution_mode,
                    "Rust expert layer streaming completed"
                );
            },
        );

        let residency_target = self.expert_residency_target(layer_index);
        let layer_has_no_retained_page =
            self.retained_expert_layers
                .as_ref()
                .is_some_and(|retained_expert_layers| {
                    retained_expert_layers
                        .borrow()
                        .retained_layer(layer_index)
                        .is_none()
                });
        let should_commit_complete_layer = route_token_count > 1
            && paged_prefill_execution_mode
                == Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
            && residency_target == Some(ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead);
        let should_commit_routed_page = route_token_count == 1
            && paged_prefill_execution_mode
                == Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
            && matches!(
                residency_target,
                Some(
                    ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead
                        | ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
                )
            )
            && layer_has_no_retained_page;
        let retained_expert_layers = self.retained_expert_layers.as_ref();
        let should_commit_mandatory_page = (should_commit_complete_layer
            || should_commit_routed_page)
            && retained_expert_layers.is_some_and(|retained_expert_layers| {
                retained_expert_layers
                    .borrow()
                    .can_commit_materialized_page(layer_index, page_manifest.payload_byte_count)
            });
        if should_commit_mandatory_page {
            let mut expert_weight_arrays = Vec::new();
            streamed_expert_weights.append_array_references(&mut expert_weight_arrays);
            let materialization_operation = if route_token_count > 1 {
                PerformanceOperation::MandatoryPrefillCompleteLayerMaterializationWait
            } else {
                PerformanceOperation::MandatoryDecodeRoutePageMaterializationWait
            };
            performance_attribution
                .measure_operation(materialization_operation, |_performance_attribution| {
                    self.runtime.evaluate_arrays(&expert_weight_arrays)
                })?;
            let expert_capacity = expert_pager.layer_plan(layer_index)?.expert_capacity;
            let retained_expert_layers =
                retained_expert_layers.ok_or_else(|| ExpertPagingError::InvalidPagingPlan {
                    description: "read-through retention lost the retained cache".to_owned(),
                })?;
            let commit_result = performance_attribution.measure_operation(
                PerformanceOperation::ExpertResidencyCommit,
                |_performance_attribution| {
                    if should_commit_complete_layer {
                        retained_expert_layers
                            .borrow_mut()
                            .commit_materialized_complete_layer(
                                layer_index,
                                expert_capacity,
                                super::super::expert_paging::expert_pager::Qwen3_5RetainedExpertLayer {
                                    weights: streamed_expert_weights,
                                    manifest: page_manifest,
                                },
                            )
                    } else {
                        retained_expert_layers
                            .borrow_mut()
                            .commit_materialized_routed_page(
                                layer_index,
                                expert_capacity,
                                streamed_expert_ids.clone(),
                                super::super::expert_paging::expert_pager::Qwen3_5RetainedExpertLayer {
                                    weights: streamed_expert_weights,
                                    manifest: page_manifest,
                                },
                            )
                    }
                },
            )
            .map_err(|commit_error| ExpertPagingError::InvalidPagingPlan {
                description: commit_error.to_string(),
            })?;
            if matches!(
                commit_result.outcome,
                RetainedExpertLayerCommitOutcome::Committed(_)
            ) {
                if let RetainedExpertLayerCommitOutcome::Committed(commit_delta) =
                    commit_result.outcome
                {
                    performance_attribution.record_counter(
                        if should_commit_complete_layer {
                            PerformanceCounter::ExpertResidencyPromotedCompletePayloadBytes
                        } else {
                            PerformanceCounter::ExpertResidencyPromotedPartialPayloadBytes
                        },
                        commit_delta.committed_payload_bytes,
                    );
                }
                let retained_cache = retained_expert_layers.borrow();
                let retained_layer =
                    retained_cache.retained_layer(layer_index).ok_or_else(|| {
                        ExpertPagingError::InvalidPagingPlan {
                            description: "committed read-through page disappeared".to_owned(),
                        }
                    })?;
                tracing::info!(
                    layer_index,
                    ?residency_target,
                    commit_outcome = ?commit_result.outcome,
                    "transferred mandatory expert read into retained ownership"
                );
                return self.forward_moe_paged_with_performance_attribution(
                    hidden_states,
                    mixture_of_experts_weights,
                    &retained_layer.weights,
                    &retained_layer.manifest,
                    selected_indices,
                    sorted_unique_expert_ids,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                    performance_attribution,
                );
            }
            let uncommitted_page = commit_result.uncommitted_page.ok_or_else(|| {
                ExpertPagingError::InvalidPagingPlan {
                    description: "rejected read-through commit lost its candidate page".to_owned(),
                }
            })?;
            if commit_result.outcome == RetainedExpertLayerCommitOutcome::RejectedByCurrentCeiling {
                performance_attribution
                    .record_counter(PerformanceCounter::ExpertResidencyCommitRejectionCount, 1);
            }
            tracing::info!(
                layer_index,
                ?residency_target,
                commit_outcome = ?commit_result.outcome,
                "kept mandatory expert page operation-local after retention declined ownership"
            );
            return self.forward_moe_paged_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &uncommitted_page.weights,
                &uncommitted_page.manifest,
                selected_indices,
                sorted_unique_expert_ids,
                selected_scores,
                should_use_compiled_elementwise_graphs,
                performance_attribution,
            );
        }

        if should_execute_token_projections_separately {
            return self.forward_moe_paged_target_verification_with_performance_attribution(
                hidden_states,
                mixture_of_experts_weights,
                &streamed_expert_weights,
                &page_manifest,
                selected_indices,
                &streamed_expert_ids,
                selected_scores,
                performance_attribution,
            );
        }
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &streamed_expert_weights,
            &page_manifest,
            selected_indices,
            sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}
