//! The router gate and top-k expert choice for one mixture-of-experts
//! forward. Where the chosen experts' weights come from differs by engine
//! (whole in memory, or paged from storage); which experts are chosen does
//! not, so the resident and streaming models share this one step.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::feed_forward_weights::{
    Qwen3_5MoEFeedForwardWeights, Qwen3_5MoERouterGateWeights,
};
use crate::qwen3_5_core::model_math::routing;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5ModelBase {
    /// Runs the router gate and top-k expert routing for one MoE forward,
    /// returning the selected expert indices and their scores.
    pub fn routed_expert_selection(
        &self,
        hidden_states: &MlxArray,
        mixture_of_experts_weights: &Qwen3_5MoEFeedForwardWeights,
    ) -> Result<(MlxArray, MlxArray), TextExecutionError> {
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
        .map_err(TextExecutionError::from)
    }
}
