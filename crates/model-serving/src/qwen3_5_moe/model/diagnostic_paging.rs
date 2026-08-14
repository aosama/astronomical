//! Explicit diagnostic paging modes kept outside production route orchestration.

use astronomical_runtime_integration::MlxArray;

use crate::PerformanceAttribution;
use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};

use super::super::expert_paging::expert_pager::Qwen3_5ExpertPager;
use super::Qwen3_5MoEPagedPrefillExecutionMode;
use super::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;

impl Qwen3_5Model {
    /// Streams the compact union of experts routed by a diagnostic prompt.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_direct_prefill_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let sorted_unique_expert_ids = self.copy_sorted_unique_expert_ids(selected_indices)?;
        let (streamed_expert_weights, page_manifest) = expert_pager
            .load_rust_streamed_expert_layer(
                &self.runtime,
                layer_index,
                &sorted_unique_expert_ids,
                performance_attribution,
            )?;
        self.forward_moe_paged_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            &streamed_expert_weights,
            &page_manifest,
            selected_indices,
            &sorted_unique_expert_ids,
            selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }

    /// Executes each prompt token through its own diagnostic routed page.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_moe_with_per_token_paging(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        expert_pager: &Qwen3_5ExpertPager,
        layer_index: usize,
        selected_indices: &MlxArray,
        selected_scores: &MlxArray,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let hidden_state_shape = hidden_states.shape();
        if hidden_state_shape.len() != 3 || hidden_state_shape[0] != 1 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "paged prefill MoE currently expects [1, tokens, hidden] states",
            });
        }
        let token_count = hidden_state_shape[1];
        let hidden_dimension = hidden_state_shape[2];
        let expert_count_per_token = selected_indices.shape().last().copied().ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "paged MoE selected indices must include an expert dimension",
            },
        )?;
        let mut token_moe_outputs = Vec::with_capacity(token_count as usize);
        for token_index in 0..token_count {
            let token_hidden_states = self.runtime.slice(
                hidden_states,
                &[0, token_index, 0],
                &[1, token_index + 1, hidden_dimension],
                &[1, 1, 1],
            )?;
            let token_selected_indices = self.runtime.slice(
                selected_indices,
                &[0, token_index, 0],
                &[1, token_index + 1, expert_count_per_token],
                &[1, 1, 1],
            )?;
            let token_selected_scores = self.runtime.slice(
                selected_scores,
                &[0, token_index, 0],
                &[1, token_index + 1, expert_count_per_token],
                &[1, 1, 1],
            )?;
            token_moe_outputs.push(self.forward_moe_with_layer_store_paging(
                &token_hidden_states,
                mixture_of_experts_weights,
                expert_pager,
                layer_index,
                token_count,
                &token_selected_indices,
                &token_selected_scores,
                should_use_compiled_elementwise_graphs,
                false,
                Qwen3_5MoEPagedPrefillExecutionMode::TokenLocalDiagnostic,
                None,
                performance_attribution,
            )?);
        }
        let token_moe_output_references = token_moe_outputs.iter().collect::<Vec<_>>();
        Ok(self
            .runtime
            .concatenate_axis(&token_moe_output_references, 1)?)
    }
}
