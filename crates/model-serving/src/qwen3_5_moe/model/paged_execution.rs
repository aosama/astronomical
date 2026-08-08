//! Paged mixture-of-experts graph execution after routing and page loading.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::super::expert_paging::expert_pager::Qwen3_5PagedExpertWeights;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination::combine_sparse_and_shared_experts;
use super::routing::{
    MINIMUM_SORTED_EXPERT_ASSIGNMENTS, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum,
};
use crate::expert_paging::QuantizedExpertPageManifest;

const REMAP_EXPERT_PAGE_SLOTS_OPERATION: &str = "remap Qwen3.5-MoE expert page slots";

impl Qwen3_5Model {
    /// Executes sparse expert computation against a compact paged weight owner.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
            &self.runtime,
            selected_indices,
            sorted_unique_expert_ids,
            page_manifest,
        )?;
        self.forward_moe_with_precomputed_paged_expert_indices(
            hidden_states,
            mixture_of_experts_weights,
            paged_expert_weights,
            &page_slot_indices,
            selected_scores,
            should_use_compiled_elementwise_graphs,
        )
    }

    fn forward_moe_with_precomputed_paged_expert_indices(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let expanded_states = self.runtime.expand_dims(hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let sorted_expert_assignments =
            if selected_expert_indices.element_count() >= MINIMUM_SORTED_EXPERT_ASSIGNMENTS {
                Some(qwen3_5_moe_sort_expert_assignments(
                    &self.runtime,
                    &expanded_states,
                    selected_expert_indices,
                )?)
            } else {
                None
            };
        let (expert_input_states, expert_indices, are_expert_indices_sorted) =
            match sorted_expert_assignments.as_ref() {
                Some((sorted_states, sorted_indices, _)) => (sorted_states, sorted_indices, true),
                None => (&expanded_states, selected_expert_indices, false),
            };
        let selected_up = self.quantized_expert_linear(
            expert_input_states,
            &paged_expert_weights.up_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let selected_gate = self.quantized_expert_linear(
            expert_input_states,
            &paged_expert_weights.gate_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.quantized_expert_linear(
            &selected_activated,
            &paged_expert_weights.down_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let sparse_expert_output = match sorted_expert_assignments.as_ref() {
            Some((_, _, inverse_order)) => qwen3_5_moe_sorted_expert_weighted_sum(
                &self.runtime,
                self.sorted_expert_weighted_sum_kernel()?,
                &selected_outputs,
                inverse_order,
                selected_scores,
            )?,
            None => {
                let selected_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
                let expanded_scores = self.runtime.expand_dims(selected_scores, -1)?;
                let weighted_outputs =
                    self.runtime.multiply(&selected_outputs, &expanded_scores)?;
                self.runtime.sum_axis(&weighted_outputs, -2, false)?
            }
        };

        let shared_gate = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_gate_projection,
        )?;
        let shared_up = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_up_projection,
        )?;
        let shared_activated =
            self.runtime
                .apply_compiled_swiglu(&self.compiled_swiglu, &shared_gate, &shared_up)?;
        let shared_output = self.quantized_linear(
            &shared_activated,
            &mixture_of_experts_weights.shared_expert_down_projection,
        )?;
        let shared_gate_logits = self.quantized_linear(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_output_gate_projection,
        )?;
        if should_use_compiled_elementwise_graphs {
            Ok(self
                .runtime
                .apply_compiled_sparse_shared_expert_combination(
                    &self.compiled_elementwise_graphs,
                    &sparse_expert_output,
                    &shared_output,
                    &shared_gate_logits,
                )?)
        } else {
            Ok(combine_sparse_and_shared_experts(
                &self.runtime,
                &sparse_expert_output,
                &shared_output,
                &shared_gate_logits,
            )?)
        }
    }

    // Complete-layer execution inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_complete_layer_paging_and_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        complete_layer_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                self.forward_moe_with_precomputed_paged_expert_indices(
                    hidden_states,
                    mixture_of_experts_weights,
                    complete_layer_expert_weights,
                    selected_indices,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                self.forward_moe_paged(
                    hidden_states,
                    mixture_of_experts_weights,
                    paged_expert_weights,
                    page_manifest,
                    selected_indices,
                    sorted_unique_expert_ids,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_target_verification_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        page_manifest: &QuantizedExpertPageManifest,
        selected_indices: &MlxArray,
        sorted_unique_expert_ids: &[usize],
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                let page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
                    &self.runtime,
                    selected_indices,
                    sorted_unique_expert_ids,
                    page_manifest,
                )?;
                self.forward_moe_target_verification_with_precomputed_paged_expert_indices(
                    hidden_states,
                    mixture_of_experts_weights,
                    paged_expert_weights,
                    &page_slot_indices,
                    selected_scores,
                )
            },
        )
    }

    pub(super) fn forward_moe_with_complete_layer_target_verification_and_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        complete_layer_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                self.forward_moe_target_verification_with_precomputed_paged_expert_indices(
                    hidden_states,
                    mixture_of_experts_weights,
                    complete_layer_expert_weights,
                    selected_indices,
                    selected_scores,
                )
            },
        )
    }

    fn forward_moe_target_verification_with_precomputed_paged_expert_indices(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        paged_expert_weights: &Qwen3_5PagedExpertWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        let batch_size = hidden_state_shape[0];
        let token_count = hidden_state_shape[1];
        let hidden_dimension = hidden_state_shape[2];
        let expert_count_per_token = selected_expert_indices.shape()[2];
        let flattened_hidden_states = self
            .runtime
            .reshape(hidden_states, &[batch_size * token_count, hidden_dimension])?;
        let flattened_expert_indices = self.runtime.reshape(
            selected_expert_indices,
            &[batch_size * token_count, expert_count_per_token],
        )?;
        let expanded_states = self.runtime.expand_dims(&flattened_hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let selected_up = self.quantized_expert_linear(
            &expanded_states,
            &paged_expert_weights.up_projection,
            &flattened_expert_indices,
            false,
        )?;
        let selected_gate = self.quantized_expert_linear(
            &expanded_states,
            &paged_expert_weights.gate_projection,
            &flattened_expert_indices,
            false,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.quantized_expert_linear(
            &selected_activated,
            &paged_expert_weights.down_projection,
            &flattened_expert_indices,
            false,
        )?;
        let selected_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
        let selected_outputs = self.runtime.reshape(
            &selected_outputs,
            &[
                batch_size,
                token_count,
                expert_count_per_token,
                hidden_dimension,
            ],
        )?;
        let expanded_scores = self.runtime.expand_dims(selected_scores, -1)?;
        let weighted_outputs = self.runtime.multiply(&selected_outputs, &expanded_scores)?;
        let sparse_expert_output = self.runtime.sum_axis(&weighted_outputs, -2, false)?;

        let execution_mode =
            crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow;
        let shared_gate = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_gate_projection,
            execution_mode,
        )?;
        let shared_up = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_up_projection,
            execution_mode,
        )?;
        let shared_activated =
            self.runtime
                .apply_compiled_swiglu(&self.compiled_swiglu, &shared_gate, &shared_up)?;
        let shared_output = self.quantized_linear_for_paged_prefill_execution_mode(
            &shared_activated,
            &mixture_of_experts_weights.shared_expert_down_projection,
            execution_mode,
        )?;
        let shared_gate_logits = self.quantized_linear_for_paged_prefill_execution_mode(
            hidden_states,
            &mixture_of_experts_weights.shared_expert_output_gate_projection,
            execution_mode,
        )?;
        Ok(combine_sparse_and_shared_experts(
            &self.runtime,
            &sparse_expert_output,
            &shared_output,
            &shared_gate_logits,
        )?)
    }

    /// Copies only the routing evidence required for CPU page selection.
    ///
    /// The expert pager needs a sorted unique host list to choose files and retained
    /// owners. The complete assignment array remains in selected_indices on MLX and is
    /// remapped later by take_axis on the graphics processor; returning the raw copied
    /// assignments here would invite the old CPU remapping path to return.
    pub(super) fn copy_sorted_unique_expert_ids(
        &self,
        selected_indices: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Vec<usize>, Qwen3_5ExecutionError> {
        let contiguous_selected_expert_ids = performance_attribution.measure_operation(
            PerformanceOperation::SelectedExpertIdContiguousGraphConstruction,
            |_performance_attribution| {
                self.runtime
                    .build_contiguous_row_major_copy(selected_indices)
            },
        )?;
        performance_attribution.measure_operation(
            PerformanceOperation::SelectedExpertIdEvaluationSynchronizationWait,
            |_performance_attribution| contiguous_selected_expert_ids.evaluate(),
        )?;
        let selected_global_expert_ids = performance_attribution.measure_operation(
            PerformanceOperation::SelectedExpertIdHostMemoryCopy,
            |_performance_attribution| contiguous_selected_expert_ids.copy_evaluated_u32_values(),
        )?;
        let mut sorted_unique_expert_ids = selected_global_expert_ids
            .iter()
            .map(|selected_global_expert_id| *selected_global_expert_id as usize)
            .collect::<Vec<_>>();
        sorted_unique_expert_ids.sort_unstable();
        sorted_unique_expert_ids.dedup();
        Ok(sorted_unique_expert_ids)
    }
}

/// Builds a lazy graphics-processor gather from global expert IDs to compact page slots.
///
/// Page loading still needs the sorted unique IDs on the CPU because those IDs select
/// files and retained expert owners. The assignment-sized selected_indices array stays
/// in MLX throughout execution. This function replaces the former per-assignment Rust
/// HashMap lookup and assignment-sized upload with one dense lookup array plus take_axis.
pub fn qwen3_5_moe_remap_expert_page_slots(
    runtime: &MlxRuntime,
    selected_indices: &MlxArray,
    sorted_unique_expert_ids: &[usize],
    page_manifest: &QuantizedExpertPageManifest,
) -> Result<MlxArray, MlxRuntimeError> {
    // The host already synchronized the router once to choose the page. Reuse that
    // bounded unique-ID evidence to fail before graph construction if the page loader
    // returned different experts. Do not synchronize selected_indices a second time.
    if sorted_unique_expert_ids != page_manifest.expert_ids {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: REMAP_EXPERT_PAGE_SLOTS_OPERATION,
            description: "routed expert IDs do not match the compact page manifest".to_owned(),
        });
    }
    let expert_capacity = i32::try_from(page_manifest.page_slot_by_global_expert_id.len())
        .map_err(|_| MlxRuntimeError::RuntimeOperation {
            operation: REMAP_EXPERT_PAGE_SLOTS_OPERATION,
            description: "expert capacity exceeds the MLX shape range".to_owned(),
        })?;

    // This table has one entry per model expert, not one entry per routed token/expert
    // assignment. For current sparse layers it is a tiny fixed-size input whose values
    // identify the compact first-dimension slot of each loaded expert tensor.
    let page_slot_by_global_expert_id = runtime.array_from_u32(
        &page_manifest.page_slot_by_global_expert_id,
        &[expert_capacity],
    )?;

    // take_axis records a lazy MLX gather on the graphics-processor stream. Its output
    // preserves the selected_indices shape, but each global expert ID is replaced by
    // the corresponding compact page slot. Later gathered quantized matrix operations
    // consume this array directly, so Rust neither loops over every assignment nor
    // constructs and uploads another assignment-sized array.
    runtime.take_axis(&page_slot_by_global_expert_id, selected_indices, 0)
}
