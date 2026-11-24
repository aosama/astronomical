use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::feed_forward_weights::Qwen3_5MoEFeedForwardWeights;
use crate::qwen3_5_resident::experts::Qwen3_5ResidentExpertLayerWeights;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::{PerformanceAttribution, PerformanceOperation};

impl Qwen3_5ResidentModel {
    pub fn forward_qwen3_5_moe(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
        resident_expert_layer_weights: &Qwen3_5ResidentExpertLayerWeights,
        should_use_compiled_elementwise_graphs: bool,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let (selected_indices, selected_scores) = performance_attribution.measure_operation(
            PerformanceOperation::ResidentMoeGraphConstruction,
            |_performance_attribution| {
                self.routed_expert_selection(hidden_states, mixture_of_experts_weights)
            },
        )?;
        self.forward_moe_resident_with_performance_attribution(
            hidden_states,
            mixture_of_experts_weights,
            resident_expert_layer_weights,
            &selected_indices,
            &selected_scores,
            should_use_compiled_elementwise_graphs,
            performance_attribution,
        )
    }
}
