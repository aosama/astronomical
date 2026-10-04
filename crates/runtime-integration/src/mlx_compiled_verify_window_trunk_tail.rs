use super::VerifyWindowInputReader;
use crate::mlx_compiled_verify_window_geometry::VerifyWindowGeometry;
use crate::mlx_compiled_verify_window_ops as ops;
use crate::{MlxArray, MlxStream, raw};

pub(super) fn trace_trunk_tail(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    reader: &VerifyWindowInputReader,
    trunk_weight_indices: &[usize; 7],
    hidden_states: &MlxArray,
) -> Result<(MlxArray, MlxArray), i32> {
    let final_normalization_weight = reader.take_at(trunk_weight_indices[3])?;
    let head_packed = reader.take_at(trunk_weight_indices[4])?;
    let head_scales = reader.take_at(trunk_weight_indices[5])?;
    let head_biases = reader.take_at(trunk_weight_indices[6])?;
    let normalized_states = ops::fast_rms_norm(
        gpu_stream,
        hidden_states,
        &final_normalization_weight,
        geometry.rms_norm_epsilon(),
    )?;
    let head_quantization = geometry.trunk_quantization().language_model_head;
    let all_position_logits = ops::quantized_matmul_affine(
        gpu_stream,
        &normalized_states,
        &head_packed,
        &head_scales,
        &head_biases,
        head_quantization.group_size,
        head_quantization.bits,
    )?;
    let logits = ops::astype(
        gpu_stream,
        &all_position_logits,
        raw::mlx_dtype__MLX_FLOAT32,
    )?;
    Ok((logits, hidden_states.retain().map_err(|_| 1)?))
}
