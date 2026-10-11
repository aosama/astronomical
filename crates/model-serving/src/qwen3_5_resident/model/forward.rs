use astronomical_mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::feed_forward_weights::{
    Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights,
};
use crate::qwen3_5_core::model_math::routing;
use crate::qwen3_5_resident::experts::Qwen3_5ResidentExpertLayerWeights;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::{PerformanceAttribution, PerformanceOperation};

impl Qwen3_5ResidentModel {
    pub(crate) fn forward_qwen3_5_moe(
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
                let router_logits = match &mixture_of_experts_weights.router_projection {
                    Qwen3_5MoERouterGateWeights::Affine(quantized_weights) => {
                        self.quantized_linear(hidden_states, quantized_weights)?
                    }
                    Qwen3_5MoERouterGateWeights::Unquantized(unquantized_weight) => {
                        let transposed_gate_weight =
                            self.runtime.transpose_axes(unquantized_weight, &[1, 0])?;
                        self.runtime
                            .matmul(hidden_states, &transposed_gate_weight)?
                    }
                };
                routing::qwen3_5_moe_route_experts(
                    &self.runtime,
                    &router_logits,
                    self.config.experts_per_token() as i32,
                    self.config.normalizes_top_k_probabilities(),
                )
                .map_err(Qwen3_5ExecutionError::from)
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
