//! Paged mixture-of-experts graph execution after routing and page loading.
//!
//! A native snapshot is more than page-table metadata: it is the lifetime owner
//! for every MLX array whose graphics-processor address appears in that table.
//! Keeping the snapshot borrowed through graph construction allows native
//! eviction to proceed without invalidating lazy or already-submitted products.

use astronomical_runtime_integration::{
    MlxArray, MlxNativeExpertCacheSnapshot, MlxNativeExpertProjection,
};

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::{PerformanceAttribution, PerformanceCounter, PerformanceOperation};

use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use super::output_combination::combine_sparse_and_shared_experts;
use super::routing::{
    MINIMUM_SORTED_EXPERT_ASSIGNMENTS, qwen3_5_moe_sort_expert_assignments,
    qwen3_5_moe_sorted_expert_weighted_sum,
};

impl Qwen3_5Model {
    /// Executes sparse expert computation directly against independently owned pages.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        native_expert_cache_snapshot: &MlxNativeExpertCacheSnapshot,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        self.forward_moe_with_precomputed_paged_expert_indices(
            hidden_states,
            mixture_of_experts_weights,
            native_expert_cache_snapshot,
            selected_indices,
            selected_scores,
            should_use_compiled_elementwise_graphs,
        )
    }

    fn forward_moe_with_precomputed_paged_expert_indices(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        native_expert_cache_snapshot: &MlxNativeExpertCacheSnapshot,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let expanded_states = self.runtime.expand_dims(hidden_states, -2)?;
        let expanded_states = self.runtime.expand_dims(&expanded_states, -3)?;
        // Sorting is worthwhile only for enough assignments to expose matrix
        // tiles. The sorted IDs let each Metal tile reuse one expert page; the
        // inverse order is consumed directly by the weighted-sum kernel so no
        // expanded token/top-K tensor needs to be restored first.
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
        // Gate, up, and down all dereference the immutable native page table.
        // The snapshot owns packed weights, affine parameters, or native BF16
        // matrices for the complete lifetime of these lazy projections.
        let selected_up = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Up,
            expert_input_states,
            expert_indices,
            true,
            are_expert_indices_sorted,
        )?;
        let selected_gate = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Gate,
            expert_input_states,
            expert_indices,
            true,
            are_expert_indices_sorted,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Down,
            &selected_activated,
            expert_indices,
            true,
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        native_expert_cache_snapshot: &MlxNativeExpertCacheSnapshot,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let paged_moe_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                self.forward_moe_paged(
                    hidden_states,
                    mixture_of_experts_weights,
                    native_expert_cache_snapshot,
                    selected_indices,
                    selected_scores,
                    should_use_compiled_elementwise_graphs,
                )
            },
        )?;
        performance_attribution
            .record_counter(PerformanceCounter::NativePagedExpertProjectionGraphCount, 3);
        Ok(paged_moe_output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_paged_target_verification_with_performance_attribution(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        native_expert_cache_snapshot: &MlxNativeExpertCacheSnapshot,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let paged_moe_output = performance_attribution.measure_operation(
            PerformanceOperation::PagedMoeGraphConstruction,
            |_performance_attribution| {
                self.forward_moe_target_verification_with_precomputed_paged_expert_indices(
                    hidden_states,
                    mixture_of_experts_weights,
                    native_expert_cache_snapshot,
                    selected_indices,
                    selected_scores,
                )
            },
        )?;
        performance_attribution
            .record_counter(PerformanceCounter::NativePagedExpertProjectionGraphCount, 3);
        Ok(paged_moe_output)
    }

    fn forward_moe_target_verification_with_precomputed_paged_expert_indices(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        native_expert_cache_snapshot: &MlxNativeExpertCacheSnapshot,
        selected_expert_indices: &MlxArray,
        selected_scores: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        // The two-token MTP verification window must preserve token order and
        // its first-row recurrent checkpoint. Flatten only for gathered expert
        // products, then restore [batch, token, top-K, hidden] before weighting.
        // This intentionally favors exact rollback semantics over sorted large-
        // prefill dispatch.
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
        let selected_up = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Up,
            &expanded_states,
            &flattened_expert_indices,
            true,
            false,
        )?;
        let selected_gate = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Gate,
            &expanded_states,
            &flattened_expert_indices,
            true,
            false,
        )?;
        let selected_activated = self.runtime.apply_compiled_swiglu(
            &self.compiled_swiglu,
            &selected_gate,
            &selected_up,
        )?;
        let selected_outputs = native_expert_cache_snapshot.gather_matmul(
            &self.runtime,
            MlxNativeExpertProjection::Down,
            &selected_activated,
            &flattened_expert_indices,
            true,
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
}
