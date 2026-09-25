//! Shared MLX arithmetic primitives for the Qwen-Image-2.1 family.
//!
//! The denoising transformer and the text encoder share two things: the artifact's 4-bit affine
//! quantization and the diffusers FP32 normalization and rotary boundaries. Both therefore live
//! here once instead of per module. Precision rules follow the references exactly: activations
//! travel in their model dtype (BF16) while every normalization and the rotary rotation compute
//! in FP32 and cast back.
//!
//! One deliberate duplication: the family's pure CPU references (`qwen_image_21::norm` for the
//! zero-centered RMSNorm, `qwen_image_21::rope` for the frequency tables, `qwen_image_21::time`
//! for the timestep sinusoid) reimplement the same formulas in f64 so hermetic tests can check the
//! runtime path without a GPU. Keep the pairs in step when either side changes: the runtime side
//! is the one the artifact sees, the CPU side is the oracle.

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxSafetensors};

use super::QwenImage21EngineError;

/// The artifact's affine quantization: 4-bit values in groups of 64 input channels.
pub(in crate::qwen_image_21) const QUANT_GROUP_SIZE: usize = 64;
pub(in crate::qwen_image_21) const QUANT_BITS: i32 = 4;
/// Four-bit values packed into each uint32 word — the divisor every packed weight's inner axis
/// carries (`in / 8`), including the token embedding whose packing runs along its output axis.
pub(in crate::qwen_image_21) const QUANT_VALUES_PER_WORD: usize = 8;

/// One 4-bit affine-quantized projection.
///
/// The reviewed artifact stores every projection as `weight` `[out, in/8]` uint32 (eight 4-bit
/// values per word) plus `scales`/`biases` `[out, in/group_size]`. MLX consumes exactly this
/// layout through `quantized_matmul_affine` with `transpose_weights = true`, so the tensors
/// load without repacking. The affine `biases` are dequantization offsets, not layer biases —
/// the reference models have no projection biases anywhere.
#[derive(Debug)]
pub(in crate::qwen_image_21) struct QuantizedLinear {
    input_width: usize,
    weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
}

impl QuantizedLinear {
    pub(in crate::qwen_image_21) fn load(
        tensors: &MlxSafetensors,
        prefix: &str,
        input_width: usize,
        output_width: usize,
    ) -> Result<Self, QwenImage21EngineError> {
        // The packed word count and group count come from the quantization scheme, not from a
        // per-tensor choice, so a mismatch here means the manifest is not the reviewed artifact.
        let weight = tensors.tensor(&format!("{prefix}.weight"))?;
        let scales = tensors.tensor(&format!("{prefix}.scales"))?;
        let biases = tensors.tensor(&format!("{prefix}.biases"))?;
        validate_shape(
            prefix,
            "weight",
            &weight,
            &[output_width, input_width / QUANT_VALUES_PER_WORD],
        )?;
        validate_shape(
            prefix,
            "scales",
            &scales,
            &[output_width, input_width / QUANT_GROUP_SIZE],
        )?;
        validate_shape(
            prefix,
            "biases",
            &biases,
            &[output_width, input_width / QUANT_GROUP_SIZE],
        )?;
        Ok(Self {
            input_width,
            weight,
            scales,
            biases,
        })
    }

    pub(in crate::qwen_image_21) fn forward(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, QwenImage21EngineError> {
        let activation_shape = activations.shape();
        if activation_shape.len() < 2
            || activation_shape[activation_shape.len() - 1] != self.input_width as i32
        {
            return Err(QwenImage21EngineError::InvalidInput {
                description: format!(
                    "projection expected trailing width {}, received {activation_shape:?}",
                    self.input_width
                ),
            });
        }
        Ok(runtime.quantized_matmul_affine(
            activations,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            QUANT_GROUP_SIZE as i32,
            QUANT_BITS,
        )?)
    }
}

/// Validates a loaded tensor's shape against the expected dimensions.
pub(in crate::qwen_image_21) fn validate_shape(
    prefix: &str,
    tensor_role: &str,
    tensor: &MlxArray,
    expected_shape: &[usize],
) -> Result<(), QwenImage21EngineError> {
    let expected_i32 = expected_shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| QwenImage21EngineError::InvalidInput {
                description: "tensor dimension exceeds the MLX integer range".to_owned(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tensor.shape() != expected_i32 {
        return Err(QwenImage21EngineError::WeightShape {
            tensor_name: format!("{prefix}.{tensor_role}"),
            actual_shape: tensor.shape(),
            expected_shape: expected_shape.to_vec(),
        });
    }
    Ok(())
}

/// `nn.LayerNorm` without affine parameters, computed in FP32.
pub(in crate::qwen_image_21) fn fp32_layer_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, QwenImage21EngineError> {
    let normalized_f32 = runtime
        .layer_norm_without_weight_and_bias(&runtime.astype(input, MlxDtype::Float32)?, epsilon)?;
    Ok(runtime.astype(&normalized_f32, input.dtype())?)
}

/// `RMSNorm` over the last axis: FP32 normalization, cast back, then the BF16 learned scale —
/// the diffusers `RMSNorm` order, which multiplies by the weight only after the cast.
pub(in crate::qwen_image_21) fn fp32_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, QwenImage21EngineError> {
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let normalized_f32 = runtime.rms_norm_without_weight(&input_f32, epsilon)?;
    let normalized = runtime.astype(&normalized_f32, input.dtype())?;
    Ok(runtime.multiply(&normalized, weight)?)
}

/// Zero-centered RMSNorm: FP32 chain `x * rrms * (weight + 1)`, then cast.
///
/// The runtime counterpart of `qwen_image_21::norm::zero_center_rms_norm`, which computes the same
/// formula on the CPU in f64 for hermetic comparison.
pub(in crate::qwen_image_21) fn fp32_zero_center_rms_norm(
    runtime: &MlxRuntime,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> Result<MlxArray, QwenImage21EngineError> {
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let normalized_f32 = runtime.rms_norm_without_weight(&input_f32, epsilon)?;
    let weight_f32 = runtime.astype(weight, MlxDtype::Float32)?;
    let unit = runtime.full(&[], 1.0, MlxDtype::Float32)?;
    let scale = runtime.add(&weight_f32, &unit)?;
    let scaled = runtime.multiply(&normalized_f32, &scale)?;
    Ok(runtime.astype(&scaled, input.dtype())?)
}

/// Applies the rotary rotation to `(B, S, H, D)` heads with adjacent real/imaginary pairs.
///
/// The reference uses the complex form (`view_as_complex` on `reshape(..., -1, 2)` multiplied by
/// the polar frequencies), which in real arithmetic is `x·cos + rotate(x)·sin` where `rotate`
/// swaps each adjacent pair to `(-imag, +real)`. The cosine/sine tables arrive as
/// `[S, D]` — every frequency repeated across its pair — and broadcast over batch and heads.
pub(in crate::qwen_image_21) fn apply_rope(
    runtime: &MlxRuntime,
    input: &MlxArray,
    cosines: &MlxArray,
    sines: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "rotary rotation expects 4D query/key heads".to_owned(),
        });
    }
    if cosines.shape() != [shape[1], shape[3]] || sines.shape() != cosines.shape() {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "rotary tables must match the token and channel counts".to_owned(),
        });
    }
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let real_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    let imaginary_values = runtime.slice(
        &input_f32,
        &[0, 0, 0, 1],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 2],
    )?;
    let negative_imaginary_values = runtime.negative(&imaginary_values)?;
    let rotated_pairs = runtime.concatenate_axis(
        &[
            &runtime.expand_dims(&negative_imaginary_values, 4)?,
            &runtime.expand_dims(&real_values, 4)?,
        ],
        4,
    )?;
    let rotated_pairs = runtime.reshape(&rotated_pairs, &shape)?;
    let broadcast_cosines = runtime.reshape(cosines, &[1, shape[1], 1, shape[3]])?;
    let broadcast_sines = runtime.reshape(sines, &[1, shape[1], 1, shape[3]])?;
    let cosine_component = runtime.multiply(&input_f32, &broadcast_cosines)?;
    let sine_component = runtime.multiply(&rotated_pairs, &broadcast_sines)?;
    let rotated = runtime.add(&cosine_component, &sine_component)?;
    Ok(runtime.astype(&rotated, input.dtype())?)
}

/// Attention scale `1 / sqrt(head_dim)`, matching the reference SDPA default.
pub(in crate::qwen_image_21) fn attention_scale(
    head_width: usize,
) -> Result<f32, QwenImage21EngineError> {
    let width =
        f32::from(
            u16::try_from(head_width).map_err(|_| QwenImage21EngineError::InvalidInput {
                description: "attention head width exceeds exact f32 range".to_owned(),
            })?,
        );
    Ok(width.sqrt().recip())
}

/// Fused scaled dot-product attention over `(B, H, S, D)` heads — the unmasked path where
/// every key is visible.
pub(in crate::qwen_image_21) fn fused_attention(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
) -> Result<MlxArray, QwenImage21EngineError> {
    Ok(runtime.scaled_dot_product_attention(queries, keys, values, scale)?)
}

/// Masked attention: scores in FP32, an additive `0 / -inf` mask over the key axis, softmax,
/// then back to the activation dtype before combining with values.
///
/// With a lower-triangular mask this is exact causal attention; the transformer additionally
/// uses it for its block-causal text segments with `[ones, tril]` masks.
pub(in crate::qwen_image_21) fn masked_attention(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    additive_mask: &MlxArray,
    scale: f32,
) -> Result<MlxArray, QwenImage21EngineError> {
    let transposed_keys = runtime.transpose_axes(keys, &[0, 1, 3, 2])?;
    let scores = runtime.matmul(queries, &transposed_keys)?;
    let scores_f32 = runtime.astype(&scores, MlxDtype::Float32)?;
    let scaled_scores = runtime.multiply_scalar(&scores_f32, scale)?;
    let masked_scores = runtime.add(&scaled_scores, additive_mask)?;
    let weights = runtime.softmax_axis(&masked_scores, -1)?;
    let weights_typed = runtime.astype(&weights, values.dtype())?;
    Ok(runtime.matmul(&weights_typed, values)?)
}

/// Applies the rotary rotation with the GPT-NeoX half-split convention to `(B, S, H, D)` heads.
///
/// The Qwen3-VL text model pairs channel `j` with channel `j + D/2` (`rotate_half`), unlike the
/// image transformer's adjacent pairs. The tables arrive as `[S, D/2]` — one entry per pair —
/// and broadcast over batch and heads: `out[j] = q[j]·cos - q[j+D/2]·sin` and
/// `out[j+D/2] = q[j+D/2]·cos + q[j]·sin`.
pub(in crate::qwen_image_21) fn apply_rope_half_split(
    runtime: &MlxRuntime,
    input: &MlxArray,
    cosines: &MlxArray,
    sines: &MlxArray,
) -> Result<MlxArray, QwenImage21EngineError> {
    let shape = input.shape();
    if shape.len() != 4 || shape[3] % 2 != 0 {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "half-split rotary rotation expects 4D heads with an even channel width"
                .to_owned(),
        });
    }
    let half_width = shape[3] / 2;
    if cosines.shape() != [shape[1], half_width] || sines.shape() != cosines.shape() {
        return Err(QwenImage21EngineError::InvalidInput {
            description: "half-split rotary tables must match the token and half-channel counts"
                .to_owned(),
        });
    }
    let input_f32 = runtime.astype(input, MlxDtype::Float32)?;
    let first_half = runtime.slice(
        &input_f32,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], shape[2], half_width],
        &[1, 1, 1, 1],
    )?;
    let second_half = runtime.slice(
        &input_f32,
        &[0, 0, 0, half_width],
        &[shape[0], shape[1], shape[2], shape[3]],
        &[1, 1, 1, 1],
    )?;
    let broadcast_cosines = runtime.reshape(cosines, &[1, shape[1], 1, half_width])?;
    let broadcast_sines = runtime.reshape(sines, &[1, shape[1], 1, half_width])?;
    let first_cosine = runtime.multiply(&first_half, &broadcast_cosines)?;
    let second_sine = runtime.multiply(&second_half, &broadcast_sines)?;
    let rotated_first = runtime.subtract(&first_cosine, &second_sine)?;
    let second_cosine = runtime.multiply(&second_half, &broadcast_cosines)?;
    let first_sine = runtime.multiply(&first_half, &broadcast_sines)?;
    let rotated_second = runtime.add(&second_cosine, &first_sine)?;
    let rotated = runtime.concatenate_axis(&[&rotated_first, &rotated_second], 3)?;
    Ok(runtime.astype(&rotated, input.dtype())?)
}
