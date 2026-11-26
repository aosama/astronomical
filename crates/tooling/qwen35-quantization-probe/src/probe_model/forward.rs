use mlx_bridge::MlxArray;

use crate::probe_model::Qwen3_5ProbeModel;
use crate::probe_model::experts::Qwen3_5ProbeExpertLayerWeights;
use crate::{PerformanceAttribution, PerformanceOperation};
use qwen35::qwen3_5_core::model_math::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5ProbeModel {
    pub fn forward_qwen3_5_moe(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        held_expert_layer_weights: &Qwen3_5ProbeExpertLayerWeights,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let (selected_indices, selected_scores) = performance_attribution.measure_operation(
            PerformanceOperation::ResidentMoeGraphConstruction,
            |_performance_attribution| {
                self.routed_expert_selection(hidden_states, mixture_of_experts_weights)
            },
        )?;
        self.forward_moe_held_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            held_expert_layer_weights,
            &selected_indices,
            &selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}
