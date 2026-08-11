//! Standard MLX gathered matrix products over complete contiguous expert arrays.
//!
//! Resident execution changes weight ownership only. It reuses the production
//! router, assignment sorting, affine profiles, SwiGLU graph, weighted sum, and
//! shared-expert combination so resident and paged modes implement the same math.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::model::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::expert_residency::Qwen3_5ResidentExpertLayerWeights;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::Qwen3_5MoEPagedPrefillExecutionMode;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination::combine_sparse_and_shared_experts;
use super::routing::{
    MINIMUM_SORTED_EXPERT_ASSIGNMENTS, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum,
};

impl Qwen3_5Model {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_resident_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        resident_expert_layer_weights: &Qwen3_5ResidentExpertLayerWeights,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::ResidentMoeGraphConstruction,
            |_performance_attribution| {
                if execution_mode == Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow {
                    self.forward_moe_resident_target_verification(
                        hidden_states,
                        mixture_of_experts_weights,
                        resident_expert_layer_weights,
                        selected_indices,
                        selected_scores,
                        execution_mode,
                    )
                } else {
                    self.forward_moe_resident(
                        hidden_states,
                        mixture_of_experts_weights,
                        resident_expert_layer_weights,
                        selected_indices,
                        selected_scores,
                        should_use_compiled_elementwise_graphs,
                        execution_mode,
                    )
                }
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_moe_resident(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        resident_expert_layer_weights: &Qwen3_5ResidentExpertLayerWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        // Add projection and expert-selection axes expected by MLX gather_mm.
        // Sorting larger assignment sets groups reads by the leading expert axis
        // and the inverse order later restores router-score alignment.
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
                Some((sorted_states, sorted_indices, _inverse_order)) => {
                    (sorted_states, sorted_indices, true)
                }
                None => (&expanded_states, selected_expert_indices, false),
            };
        let selected_up = self.resident_expert_linear(
            expert_input_states,
            &resident_expert_layer_weights.up_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let selected_gate = self.resident_expert_linear(
            expert_input_states,
            &resident_expert_layer_weights.gate_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.resident_expert_linear(
            &selected_activated,
            &resident_expert_layer_weights.down_projection,
            expert_indices,
            are_expert_indices_sorted,
        )?;
        let sparse_expert_output = match sorted_expert_assignments.as_ref() {
            Some((_sorted_states, _sorted_indices, inverse_order)) => {
                qwen3_5_moe_sorted_expert_weighted_sum(
                    &self.runtime,
                    self.sorted_expert_weighted_sum_kernel()?,
                    &selected_outputs,
                    inverse_order,
                    selected_scores,
                )?
            }
            None => {
                let selected_outputs = self.runtime.squeeze_axis(&selected_outputs, -2)?;
                let expanded_scores = self.runtime.expand_dims(selected_scores, -1)?;
                let weighted_outputs =
                    self.runtime.multiply(&selected_outputs, &expanded_scores)?;
                self.runtime.sum_axis(&weighted_outputs, -2, false)?
            }
        };
        self.combine_resident_sparse_and_shared_experts(
            hidden_states,
            mixture_of_experts_weights,
            &sparse_expert_output,
            should_use_compiled_elementwise_graphs,
            execution_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_moe_resident_target_verification(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        resident_expert_layer_weights: &Qwen3_5ResidentExpertLayerWeights,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        let batch_size = hidden_state_shape[0];
        let token_count = hidden_state_shape[1];
        let hidden_dimension = hidden_state_shape[2];
        let expert_count_per_token = selected_expert_indices.shape()[2];
        // MTP verification keeps token rows explicit and unsorted because the
        // first verified row is also the exact rollback boundary after rejection.
        let flattened_hidden_states = self
            .runtime
            .reshape(hidden_states, &[batch_size * token_count, hidden_dimension])?;
        let flattened_expert_indices = self.runtime.reshape(
            selected_expert_indices,
            &[batch_size * token_count, expert_count_per_token],
        )?;
        let expanded_states = self.runtime.expand_dims(&flattened_hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        let selected_up = self.resident_expert_linear(
            &expanded_states,
            &resident_expert_layer_weights.up_projection,
            &flattened_expert_indices,
            false,
        )?;
        let selected_gate = self.resident_expert_linear(
            &expanded_states,
            &resident_expert_layer_weights.gate_projection,
            &flattened_expert_indices,
            false,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = self.resident_expert_linear(
            &selected_activated,
            &resident_expert_layer_weights.down_projection,
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
        self.combine_resident_sparse_and_shared_experts(
            hidden_states,
            mixture_of_experts_weights,
            &sparse_expert_output,
            false,
            execution_mode,
        )
    }

    fn resident_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
        selected_expert_indices: &MlxArray,
        are_expert_indices_sorted: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        // Both variants select directly from complete arrays with original
        // expert IDs. Quantized companions retain their independent source dtype.
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                let transposed_expert_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(self.runtime.gather_dense_matmul(
                    activations,
                    &transposed_expert_weights,
                    None,
                    Some(selected_expert_indices),
                    are_expert_indices_sorted,
                )?)
            }
            Qwen3_5AffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => Ok(self.runtime.gather_quantized_matmul_affine(
                activations,
                packed_weight,
                quantization_scales,
                quantization_biases,
                None,
                Some(selected_expert_indices),
                true,
                *quantization_group_size,
                *quantization_bits,
                are_expert_indices_sorted,
            )?),
        }
    }

    fn combine_resident_sparse_and_shared_experts(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        sparse_expert_output: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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
        if should_use_compiled_elementwise_graphs {
            Ok(self
                .runtime
                .apply_compiled_sparse_shared_expert_combination(
                    &self.compiled_elementwise_graphs,
                    sparse_expert_output,
                    &shared_output,
                    &shared_gate_logits,
                )?)
        } else {
            Ok(combine_sparse_and_shared_experts(
                &self.runtime,
                sparse_expert_output,
                &shared_output,
                &shared_gate_logits,
            )?)
        }
    }
}
