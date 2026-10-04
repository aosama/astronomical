use super::VerifyWindowInputReader;
use crate::mlx_compiled_verify_window_geometry::{
    VerifyWindowAffineSlot, VerifyWindowGeometry, VerifyWindowInputSlot,
    VerifyWindowQuantizationPair, VerifyWindowTrunkWeightSlot, verify_window_input_slots,
};
use crate::mlx_compiled_verify_window_ops as ops;

use astronomical_mlx_c_rust::{MlxArray, MlxStream, raw};

#[allow(clippy::type_complexity)]
pub(super) fn trace_embedding_and_header(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    input_vector: &raw::mlx_vector_array,
) -> Result<(MlxArray, [usize; 7]), i32> {
    let token_indices = ops::builder_input(*input_vector, 0)?;
    let plan = verify_window_input_slots(geometry);
    let slot_index = |slot: &VerifyWindowInputSlot| -> Result<usize, i32> {
        plan.iter().position(|plan_slot| plan_slot == slot).ok_or(1)
    };
    let embedding_packed_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::PackedWeight),
    ))?;
    let embedding_scales_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::Scales),
    ))?;
    let embedding_biases_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::Embedding(VerifyWindowAffineSlot::Biases),
    ))?;
    let final_normalization_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::FinalNormalization,
    ))?;
    let head_packed_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::PackedWeight),
    ))?;
    let head_scales_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::Scales),
    ))?;
    let head_biases_index = slot_index(&VerifyWindowInputSlot::TrunkWeight(
        VerifyWindowTrunkWeightSlot::LanguageModelHead(VerifyWindowAffineSlot::Biases),
    ))?;
    let selected_weights = ops::take_axis_zero(
        gpu_stream,
        &ops::builder_input(*input_vector, embedding_packed_index)?,
        &token_indices,
    )?;
    let selected_scales = ops::take_axis_zero(
        gpu_stream,
        &ops::builder_input(*input_vector, embedding_scales_index)?,
        &token_indices,
    )?;
    let selected_biases = ops::take_axis_zero(
        gpu_stream,
        &ops::builder_input(*input_vector, embedding_biases_index)?,
        &token_indices,
    )?;
    let embedding_quantization = geometry.trunk_quantization().embedding;
    let hidden_states = ops::dequantize_affine(
        gpu_stream,
        &selected_weights,
        &selected_scales,
        &selected_biases,
        embedding_quantization.group_size,
        embedding_quantization.bits,
    )?;
    Ok((
        hidden_states,
        [
            embedding_packed_index,
            embedding_scales_index,
            embedding_biases_index,
            final_normalization_index,
            head_packed_index,
            head_scales_index,
            head_biases_index,
        ],
    ))
}

pub(super) fn take_affine(
    reader: &mut VerifyWindowInputReader,
) -> Result<(MlxArray, MlxArray, MlxArray), i32> {
    let packed_weight = reader.take()?;
    let scales = reader.take()?;
    let biases = reader.take()?;
    Ok((packed_weight, scales, biases))
}

pub(super) fn quantized_matmul(
    gpu_stream: &MlxStream,
    quantization: VerifyWindowQuantizationPair,
    activations: &MlxArray,
    affine: &(MlxArray, MlxArray, MlxArray),
) -> Result<MlxArray, i32> {
    ops::quantized_matmul_affine(
        gpu_stream,
        activations,
        &affine.0,
        &affine.1,
        &affine.2,
        quantization.group_size,
        quantization.bits,
    )
}

/// The dense SwiGLU feed-forward and both residual adds, shared by every
/// layer family: `residual + down(silu(gate(norm2)) * up(norm2))`.
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_feed_forward_tail(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    reader: &mut VerifyWindowInputReader,
    feed_forward_gate: VerifyWindowQuantizationPair,
    feed_forward_up: VerifyWindowQuantizationPair,
    feed_forward_down: VerifyWindowQuantizationPair,
    attention_residual: MlxArray,
) -> Result<MlxArray, i32> {
    let post_attention_normalization_weight = reader.take()?;
    let mlp_gate = take_affine(reader)?;
    let mlp_up = take_affine(reader)?;
    let mlp_down = take_affine(reader)?;
    let normalized_attention = ops::fast_rms_norm(
        gpu_stream,
        &attention_residual,
        &post_attention_normalization_weight,
        geometry.rms_norm_epsilon(),
    )?;
    let gate_activations = quantized_matmul(
        gpu_stream,
        feed_forward_gate,
        &normalized_attention,
        &mlp_gate,
    )?;
    let up_activations =
        quantized_matmul(gpu_stream, feed_forward_up, &normalized_attention, &mlp_up)?;
    let sigmoid_gate = ops::sigmoid(gpu_stream, &gate_activations)?;
    let activated_gate = ops::multiply(gpu_stream, &gate_activations, &sigmoid_gate)?;
    let intermediate = ops::multiply(gpu_stream, &activated_gate, &up_activations)?;
    let down_activations =
        quantized_matmul(gpu_stream, feed_forward_down, &intermediate, &mlp_down)?;
    ops::add(gpu_stream, &attention_residual, &down_activations)
}
