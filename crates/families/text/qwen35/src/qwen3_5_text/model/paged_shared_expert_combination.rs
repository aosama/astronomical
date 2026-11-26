//! A paged expert layer's last step: the routed experts' router-weighted
//! sum joined with the gated shared expert.
//!
//! A prompt chunk whose assignments were sorted does both in one Metal pass
//! with the fused kernel, bit for bit equal to reducing first and combining
//! after (issue #2585): the kernel reads only the routed rows, so it serves
//! experts held in the `astro_expert_cache` and experts streamed from the
//! SSD alike. One reply token, unsorted assignments and dtypes the kernel
//! is not proven for reduce first and then combine, as before.

use mlx_bridge::MlxArray;
use text_generation::execution_error::TextExecutionError;

use crate::qwen3_5_core::model_math::sorted_expert_shared_combination::{
    SortedExpertSharedCombinationInputs, fused_sorted_expert_shared_combination_if_proven,
};
use crate::qwen3_5_text::model::Qwen3_5Model;

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination;
use super::paged_execution::RoutedExpertAssignmentOutputs;
use super::routing::{
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_unsorted_expert_weighted_sum,
};

/// The shared expert's down-projection output and its gate logits.
struct SharedExpertOutputs {
    output: MlxArray,
    gate_logits: MlxArray,
}

impl Qwen3_5Model {
    /// Joins `(routed rows, router scores)` with the shared expert.
    pub(super) fn combine_routed_assignments_and_shared_outputs(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        (routed_assignment_outputs, selected_scores): (&RoutedExpertAssignmentOutputs, &MlxArray),
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, TextExecutionError> {
        let shared_expert =
            self.shared_expert_outputs(hidden_states, mixture_of_experts_weights)?;
        if should_use_compiled_elementwise_graphs
            && let Some(fused_output) = self.fused_routed_and_shared_combination(
                routed_assignment_outputs,
                selected_scores,
                &shared_expert,
            )?
        {
            return Ok(fused_output);
        }
        let sparse_expert_output =
            self.reduced_routed_assignment_outputs(routed_assignment_outputs, selected_scores)?;
        self.combine_reduced_and_shared_outputs(
            &sparse_expert_output,
            &shared_expert,
            should_use_compiled_elementwise_graphs,
        )
    }

    /// Joins an already reduced routed sum with the shared expert: the
    /// reply token's mixed route and the prediction head reduce their own.
    pub(super) fn combine_paged_sparse_and_shared_outputs(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        sparse_expert_output: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, TextExecutionError> {
        let shared_expert =
            self.shared_expert_outputs(hidden_states, mixture_of_experts_weights)?;
        self.combine_reduced_and_shared_outputs(
            sparse_expert_output,
            &shared_expert,
            should_use_compiled_elementwise_graphs,
        )
    }

    /// The router-score weighted sum of the routed rows, through the sorted
    /// kernel when the assignments were sorted.
    pub(super) fn reduced_routed_assignment_outputs(
        &self,
        routed_assignment_outputs: &RoutedExpertAssignmentOutputs,
        selected_scores: &MlxArray,
    ) -> Result<MlxArray, TextExecutionError> {
        match routed_assignment_outputs {
            RoutedExpertAssignmentOutputs::Sorted {
                sorted_outputs,
                inverse_order,
            } => Ok(qwen3_5_moe_sorted_expert_weighted_sum(
                &self.runtime,
                self.sorted_expert_weighted_sum_kernel()?,
                sorted_outputs,
                inverse_order,
                selected_scores,
            )?),
            RoutedExpertAssignmentOutputs::Unsorted { assignment_outputs } => {
                Ok(qwen3_5_moe_unsorted_expert_weighted_sum(
                    &self.runtime,
                    assignment_outputs,
                    selected_scores,
                )?)
            }
        }
    }

    /// The fused one-pass combination, or `None` when it does not apply:
    /// the kernel was not kept at load, the assignments were not sorted, or
    /// the dtypes are not the ones its bit-for-bit contract is proven for.
    fn fused_routed_and_shared_combination(
        &self,
        routed_assignment_outputs: &RoutedExpertAssignmentOutputs,
        selected_scores: &MlxArray,
        shared_expert: &SharedExpertOutputs,
    ) -> Result<Option<MlxArray>, TextExecutionError> {
        let (
            Some(combination_kernel),
            RoutedExpertAssignmentOutputs::Sorted {
                sorted_outputs,
                inverse_order,
            },
        ) = (
            self.optional_accelerators
                .sorted_expert_shared_combination_kernel
                .as_ref(),
            routed_assignment_outputs,
        )
        else {
            return Ok(None);
        };
        let combination_inputs = SortedExpertSharedCombinationInputs {
            sorted_expert_outputs: sorted_outputs,
            inverse_order,
            selected_scores,
            shared_expert_output: &shared_expert.output,
            shared_expert_gate_logits: &shared_expert.gate_logits,
        };
        Ok(fused_sorted_expert_shared_combination_if_proven(
            &self.runtime,
            combination_kernel,
            &combination_inputs,
        )?)
    }

    fn shared_expert_outputs(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
    ) -> Result<SharedExpertOutputs, TextExecutionError> {
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
        Ok(SharedExpertOutputs {
            output: self.quantized_linear(
                &shared_activated,
                &mixture_of_experts_weights.shared_expert_down_projection,
            )?,
            gate_logits: self.quantized_linear(
                hidden_states,
                &mixture_of_experts_weights.shared_expert_output_gate_projection,
            )?,
        })
    }

    fn combine_reduced_and_shared_outputs(
        &self,
        sparse_expert_output: &MlxArray,
        shared_expert: &SharedExpertOutputs,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, TextExecutionError> {
        if should_use_compiled_elementwise_graphs {
            return Ok(self
                .runtime
                .apply_compiled_sparse_shared_expert_combination(
                    &self.compiled_elementwise_graphs,
                    sparse_expert_output,
                    &shared_expert.output,
                    &shared_expert.gate_logits,
                )?);
        }
        Ok(output_combination::combine_sparse_and_shared_experts(
            &self.runtime,
            sparse_expert_output,
            &shared_expert.output,
            &shared_expert.gate_logits,
        )?)
    }
}
