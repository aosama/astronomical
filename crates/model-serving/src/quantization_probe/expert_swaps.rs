//! The parts of a mixture-of-experts layer a probe swaps that are not plain
//! affine weights in the decoder layer: the router, and a routed-expert gate
//! or up projection, which the resident engine may hold fused into one array.
//!
//! A probe re-encodes one projection at a time. The resident owner fuses the
//! gate and up projections along the output axis when their storage agrees,
//! so that one gathered multiply serves both; a probed half no longer agrees
//! with its sibling (a different width, or re-rounded values that must not
//! leak into the other projection's measurement). The swap therefore hands
//! the layer a separate pair for the duration of the probe, which is the
//! same representation production uses for a mixed-width pair, and restores
//! the owner's original value afterwards.
//!
//! OptiQ's counterpart is `_mutate_quantized_layer_to_bits` on a
//! `QuantizedSwitchLinear` (`optiq/core/sensitivity.py`); mlx-lm keeps the
//! three projections apart, so it has no fused case.

use astronomical_quantization::error::QuantizationError;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;
use crate::qwen3_5_core::model_math::feed_forward_weights::Qwen3_5MoERouterGateWeights;
use crate::qwen3_5_resident::experts::Qwen3_5ResidentGateUpWeights;

use super::simulated_quantization::mlx_step;

/// Why a probed layer's gate and up projections are held apart.
pub(crate) const PROBE_SEPARATION_REASON: &str =
    "one projection is swapped by the sensitivity probe";

/// What a swap took out of the model, kept until the restore.
#[derive(Debug)]
pub(crate) enum SwappedOutWeights {
    Affine(Qwen3_5AffineWeights),
    Router(Qwen3_5MoERouterGateWeights),
    ExpertGateUp(Qwen3_5ResidentGateUpWeights),
}

/// The router's weight in the affine form a probe re-encodes. A bfloat16
/// router is one plain array; the probe treats it like any bf16 module.
pub(crate) fn router_as_affine_weights(
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

/// The gate and up projections of one layer as two separate weights,
/// splitting the owner's fused array when it holds one.
pub(crate) fn separate_gate_and_up_projections(
    runtime: &MlxRuntime,
    gate_up_weights: &Qwen3_5ResidentGateUpWeights,
) -> Result<(Qwen3_5AffineWeights, Qwen3_5AffineWeights), QuantizationError> {
    match gate_up_weights {
        Qwen3_5ResidentGateUpWeights::Separate {
            gate_projection,
            up_projection,
            ..
        } => Ok((
            mlx_step(
                gate_projection.retained_reference(),
                "retain the gate projection",
            )?,
            mlx_step(
                up_projection.retained_reference(),
                "retain the up projection",
            )?,
        )),
        Qwen3_5ResidentGateUpWeights::Fused { projection, .. } => Ok((
            output_rows_half(runtime, projection, 0)?,
            output_rows_half(runtime, projection, 1)?,
        )),
    }
}

/// The lower (`half_index` 0, the gate) or upper (1, the up) half of a fused
/// `[experts, 2 * intermediate, ...]` weight along its output-row axis.
/// Groups run along the last axis, so a packed weight's words, scales and
/// biases all split on the same rows and each half stays a valid encoding.
fn output_rows_half(
    runtime: &MlxRuntime,
    fused_projection: &Qwen3_5AffineWeights,
    half_index: i32,
) -> Result<Qwen3_5AffineWeights, QuantizationError> {
    match fused_projection {
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
            Ok(Qwen3_5AffineWeights::NativeBfloat16 {
                weight: half_of_axis_one(runtime, weight, half_index)?,
            })
        }
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            quantization_bits,
            quantization_group_size,
        } => Ok(Qwen3_5AffineWeights::Quantized {
            packed_weight: half_of_axis_one(runtime, packed_weight, half_index)?,
            quantization_scales: half_of_axis_one(runtime, quantization_scales, half_index)?,
            quantization_biases: half_of_axis_one(runtime, quantization_biases, half_index)?,
            quantization_bits: *quantization_bits,
            quantization_group_size: *quantization_group_size,
        }),
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => Err(QuantizationError::ProbeModelFailure {
            description: "codebook-packed routed experts cannot be probed".to_owned(),
        }),
    }
}

/// One half of an array along axis 1, evaluated so the probe's forwards do
/// not re-slice the fused array on every call.
fn half_of_axis_one(
    runtime: &MlxRuntime,
    fused_array: &MlxArray,
    half_index: i32,
) -> Result<MlxArray, QuantizationError> {
    let fused_shape = fused_array.shape();
    if fused_shape.len() < 2 || fused_shape[1] % 2 != 0 {
        return Err(QuantizationError::ProbeModelFailure {
            description: format!(
                "a fused gate-up array of shape {fused_shape:?} does not split into two halves"
            ),
        });
    }
    let half_extent = fused_shape[1] / 2;
    let mut starts = vec![0_i32; fused_shape.len()];
    let mut stops = fused_shape.clone();
    starts[1] = half_extent * half_index;
    stops[1] = starts[1] + half_extent;
    let strides = vec![1_i32; fused_shape.len()];
    let half = mlx_step(
        runtime.slice(fused_array, &starts, &stops, &strides),
        "split a fused gate-up array",
    )?;
    mlx_step(half.evaluate(), "evaluate a gate-up half")?;
    Ok(half)
}
