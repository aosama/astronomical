//! The replacement weights a probe swaps in, and the float32 KL it measures.
//!
//! The bf16 reference path quantizes and immediately dequantizes the
//! module's own bf16 weight, so the forward sees exactly the rounding the
//! artifact would carry while the kernels stay the bf16 ones. The
//! uniform-baseline path re-encodes the module from its streamed bf16
//! source at the probed width and swaps in real packed weights, because
//! the loaded model is itself quantized.
//!
//! The quantize-dequantize swap and the KL are ported from OptiQ's `_exact_with_bf16_reference` and `_kl_from_ref` (`optiq/core/sensitivity.py`).

use astronomical_quantization::bit_width::{BitWidth, GroupSize};
use astronomical_quantization::error::QuantizationError;
use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AffineWeights;

/// Turns any MLX failure into the probe error the sweep understands.
pub(crate) fn mlx_step<Value, Failure: std::fmt::Display>(
    outcome: Result<Value, Failure>,
    operation: &'static str,
) -> Result<Value, QuantizationError> {
    match outcome {
        Ok(value) => Ok(value),
        Err(failure) => Err(QuantizationError::MlxRuntimeFailure {
            operation,
            description: failure.to_string(),
        }),
    }
}

/// bf16 path: `dequantize(quantize(weight))` cast back to bfloat16.
pub(crate) fn quantize_dequantized_bfloat16(
    runtime: &MlxRuntime,
    current_weights: &Qwen3_5AffineWeights,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<Qwen3_5AffineWeights, QuantizationError> {
    let Qwen3_5AffineWeights::NativeBfloat16 { weight } = current_weights else {
        return Err(QuantizationError::ProbeModelFailure {
            description: "the bf16 reference path found a module that is not bf16".to_owned(),
        });
    };
    let (group_elements, bits) = (group_size.elements() as i32, bit_width.bits() as i32);
    let (packed_weight, scales, biases) = mlx_step(
        runtime.quantize_affine(weight, group_elements, bits),
        "simulate-quantize a probed module",
    )?;
    let reconstruction = mlx_step(
        runtime.dequantize_affine(&packed_weight, &scales, &biases, group_elements, bits),
        "dequantize a probed module",
    )?;
    let bfloat16_weight = mlx_step(
        runtime.astype(&reconstruction, MlxDtype::BFloat16),
        "cast a probed module back to bf16",
    )?;
    mlx_step(bfloat16_weight.evaluate(), "evaluate a probed module")?;
    Ok(Qwen3_5AffineWeights::NativeBfloat16 {
        weight: bfloat16_weight,
    })
}

/// Uniform-baseline path: the streamed bf16 source encoded at the probed
/// width, as the artifact would store it.
pub(crate) fn encoded_from_source(
    runtime: &MlxRuntime,
    source_weight: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<Qwen3_5AffineWeights, QuantizationError> {
    if bit_width.keeps_bfloat16() {
        return Err(QuantizationError::KeepBfloat16TierNeedsBfloat16Reference);
    }
    let (group_elements, bits) = (group_size.elements() as i32, bit_width.bits() as i32);
    let (packed_weight, quantization_scales, quantization_biases) = mlx_step(
        runtime.quantize_affine(source_weight, group_elements, bits),
        "encode a probed module from its bf16 source",
    )?;
    mlx_step(
        runtime.evaluate_arrays(&[&packed_weight, &quantization_scales, &quantization_biases]),
        "evaluate a probed module",
    )?;
    Ok(Qwen3_5AffineWeights::Quantized {
        packed_weight,
        quantization_scales,
        quantization_biases,
        quantization_bits: bits,
        quantization_group_size: group_elements,
    })
}

/// Mean over positions of `sum_v p_ref * (log p_ref - log p_probe)`, in
/// float32 on device. bf16 log-softmax over a 248k vocabulary is off by
/// more than a probe's whole signal, so both sides are widened first.
pub(crate) fn mean_kl_divergence(
    runtime: &MlxRuntime,
    reference_logits: &MlxArray,
    probe_logits: &MlxArray,
) -> Result<f64, QuantizationError> {
    let per_position_kl = per_position_kl_array(runtime, reference_logits, probe_logits)?;
    let mean_kl = mlx_step(
        runtime.mean(&per_position_kl, false),
        "average over positions",
    )?;
    let mean_kl_values = mlx_step(mean_kl.to_vec_f32(), "read the KL back")?;
    Ok(f64::from(
        mean_kl_values.first().copied().unwrap_or(f32::NAN),
    ))
}

/// The same divergence at every position, read back to the host: what a
/// whole-artifact evaluation takes its median and tail from.
pub(crate) fn per_position_kl_divergences(
    runtime: &MlxRuntime,
    reference_logits: &MlxArray,
    probe_logits: &MlxArray,
) -> Result<Vec<f64>, QuantizationError> {
    let per_position_kl = per_position_kl_array(runtime, reference_logits, probe_logits)?;
    let per_position_values = mlx_step(per_position_kl.to_vec_f32(), "read the KL back")?;
    let mut divergences = Vec::with_capacity(per_position_values.len());
    for per_position_value in per_position_values {
        divergences.push(f64::from(per_position_value));
    }
    Ok(divergences)
}

/// `sum_v p_ref * (log p_ref - log p_probe)` at every position, on device.
fn per_position_kl_array(
    runtime: &MlxRuntime,
    reference_logits: &MlxArray,
    probe_logits: &MlxArray,
) -> Result<MlxArray, QuantizationError> {
    let reference_log_probabilities = float32_log_softmax(runtime, reference_logits)?;
    let probe_log_probabilities = float32_log_softmax(runtime, probe_logits)?;
    let reference_probabilities = mlx_step(
        runtime.exp(&reference_log_probabilities),
        "exponentiate the reference",
    )?;
    let log_ratio = mlx_step(
        runtime.subtract(&reference_log_probabilities, &probe_log_probabilities),
        "subtract log probabilities",
    )?;
    mlx_step(
        runtime.sum_axis(
            &mlx_step(
                runtime.multiply(&reference_probabilities, &log_ratio),
                "weight the log ratio",
            )?,
            -1,
            false,
        ),
        "sum over the vocabulary",
    )
}

fn float32_log_softmax(
    runtime: &MlxRuntime,
    logits: &MlxArray,
) -> Result<MlxArray, QuantizationError> {
    let float32_logits = mlx_step(runtime.astype(logits, MlxDtype::Float32), "widen logits")?;
    let log_normalizer = mlx_step(
        runtime.logsumexp_axis(&float32_logits, -1, true),
        "normalize logits",
    )?;
    mlx_step(
        runtime.subtract(&float32_logits, &log_normalizer),
        "subtract the normalizer",
    )
}
