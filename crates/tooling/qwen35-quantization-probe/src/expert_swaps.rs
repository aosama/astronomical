//! The parts of a mixture-of-experts layer a probe swaps that are not plain
//! affine weights in the decoder layer: the router, and a routed-expert
//! projection across every expert of a layer, which the chat model holds in
//! its `astro_expert_cache` table for that layer.
//!
//! A probe re-encodes one projection at a time and serving keeps the gate,
//! up and down projections apart, so each is replaced on its own, at any
//! width, and its sibling projections keep their own values.
//!
//! OptiQ's counterpart is `_mutate_quantized_layer_to_bits` on a
//! `QuantizedSwitchLinear` (`optiq/core/sensitivity.py`).

use quantization::error::QuantizationError;

use qwen35::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use qwen35::qwen3_5_core::model_math::feed_forward_weights::Qwen3_5MoERouterGateWeights;

use super::simulated_quantization::mlx_step;

/// What a swap took out of the model, kept until the restore.
#[derive(Debug)]
pub enum SwappedOutWeights {
    Affine(Qwen3_5AffineWeights),
    Router(Qwen3_5MoERouterGateWeights),
    RoutedExpert(Qwen3_5AffineWeights),
}

/// The router's weight in the affine form a probe re-encodes. A bfloat16
/// router is one plain array; the probe treats it like any bf16 module.
pub fn router_as_affine_weights(
    router_weights: &Qwen3_5MoERouterGateWeights,
) -> Result<Qwen3_5AffineWeights, QuantizationError> {
    match router_weights {
        Qwen3_5MoERouterGateWeights::Affine(affine_weights) => {
            mlx_step(affine_weights.retained_reference(), "retain the router")
        }
        Qwen3_5MoERouterGateWeights::Unquantized(unquantized_weight) => {
            Ok(Qwen3_5AffineWeights::NativeBfloat16 {
                weight: mlx_step(unquantized_weight.retain(), "retain the router")?,
            })
        }
    }
}
