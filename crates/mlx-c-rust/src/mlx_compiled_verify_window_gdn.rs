//! Gated-delta layer tracing for the compiled MTP verification window.
//!
//! One layer = input normalization, four quantized projections, the decay
//! formula, one recurrence implementation (fused Metal kernels when the
//! model retained them and the geometry fits, composed MLX ops otherwise),
//! then output gating, projection, and the shared feed-forward tail. The
//! layer emits its next rolling and recurrent states plus per-row boundary
//! snapshots for verifier-prefix rollback.

use super::trunk;
use super::{VerifyWindowGdnKernelSet, VerifyWindowInputReader};
use crate::mlx_compiled_verify_window_geometry::VerifyWindowGeometry;
use crate::mlx_compiled_verify_window_ops as ops;

use crate::{MlxArray, MlxDtype, MlxStream, raw};

#[path = "mlx_compiled_verify_window_gdn_recurrence.rs"]
mod recurrence;

fn can_trace_fused_gated_delta(
    kernels: VerifyWindowGdnKernelSet<'_>,
    row_count: i32,
    head_dimension: i32,
    mixed_queries_keys_values: &MlxArray,
    recurrent_state: &MlxArray,
) -> bool {
    kernels.prework_kernel.is_some()
        && kernels.checkpoint_kernel.is_some()
        && kernels.prework_lane_count > 0
        && kernels.checkpoint_threadgroup_thread_count > 0
        && kernels.checkpoint_value_row_block_size > 0
        && (2..=4).contains(&row_count)
        && head_dimension == 128
        && mixed_queries_keys_values.dtype() == MlxDtype::BFloat16
        && recurrent_state.dtype() == MlxDtype::Float32
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::type_complexity)]
pub(super) fn trace_gated_delta_layer(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    gdn_kernels: VerifyWindowGdnKernelSet<'_>,
    reader: &mut VerifyWindowInputReader,
    input_vector: &raw::mlx_vector_array,
    layer_index: usize,
    hidden_states: MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray, Vec<MlxArray>, Vec<MlxArray>), i32> {
    let row_count = geometry.row_count();
    let linear_head_dimension = geometry.linear_head_dimension();
    let linear_value_head_count = geometry.linear_value_head_count();
    let epsilon = geometry.rms_norm_epsilon();
    let layer_quantization = geometry
        .layer_quantization(layer_index)
        .and_then(|layer_quantization| layer_quantization.gated_delta)
        .ok_or(1)?;

    let rolling_state = reader.take()?;
    let recurrent_state = reader.take()?;
    let input_normalization_weight = reader.take()?;
    let input_queries_keys_values = trunk::take_affine(reader)?;
    let output_gate_projection = trunk::take_affine(reader)?;
    let update_rate_projection = trunk::take_affine(reader)?;
    let decay_interval_projection = trunk::take_affine(reader)?;
    let convolution_weight = reader.take()?;
    let decay_interval_bias = reader.take()?;
    let decay_rate_logarithm = reader.take()?;
    let normalization_weight = reader.take()?;
    let output_projection = trunk::take_affine(reader)?;

    let query_normalization_scale = ops::builder_input(*input_vector, 3)?;
    let key_normalization_scale = ops::builder_input(*input_vector, 4)?;

    let normalized_input = ops::fast_rms_norm(
        gpu_stream,
        &hidden_states,
        &input_normalization_weight,
        epsilon,
    )?;
    let mixed_queries_keys_values = trunk::quantized_matmul(
        gpu_stream,
        layer_quantization.input_queries_keys_values,
        &normalized_input,
        &input_queries_keys_values,
    )?;
    let output_gate_logits = trunk::quantized_matmul(
        gpu_stream,
        layer_quantization.output_gate,
        &normalized_input,
        &output_gate_projection,
    )?;
    let output_gate = ops::reshape(
        gpu_stream,
        &output_gate_logits,
        &[1, row_count, linear_value_head_count, linear_head_dimension],
    )?;
    let update_logits = trunk::quantized_matmul(
        gpu_stream,
        layer_quantization.update_rate,
        &normalized_input,
        &update_rate_projection,
    )?;
    let decay_inputs = trunk::quantized_matmul(
        gpu_stream,
        layer_quantization.decay_interval,
        &normalized_input,
        &decay_interval_projection,
    )?;

    let (update_rates, decays) = recurrence::trace_decays_and_rates(
        gpu_stream,
        &update_logits,
        &decay_inputs,
        &decay_interval_bias,
        &decay_rate_logarithm,
    )?;
    let convolution_input =
        ops::concatenate_axis(gpu_stream, &[&rolling_state, &mixed_queries_keys_values], 1)?;
    let (
        sequence_output,
        next_rolling_state,
        current_recurrent_state,
        boundary_convolution_states,
        boundary_recurrent_states,
    ) = if can_trace_fused_gated_delta(
        gdn_kernels,
        row_count,
        linear_head_dimension,
        &mixed_queries_keys_values,
        &recurrent_state,
    ) {
        recurrence::trace_fused_gated_delta(
            gpu_stream,
            gdn_kernels,
            geometry,
            &rolling_state,
            &convolution_input,
            &recurrent_state,
            &mixed_queries_keys_values,
            &convolution_weight,
            gdn_kernels.query_normalization_scale,
            gdn_kernels.key_normalization_scale,
            &update_rates,
            &decays,
        )?
    } else {
        recurrence::trace_composed_gated_delta(
            gpu_stream,
            geometry,
            &convolution_input,
            &recurrent_state,
            &convolution_weight,
            &query_normalization_scale,
            &key_normalization_scale,
            &update_rates,
            &decays,
        )?
    };
    let normalized_output =
        ops::fast_rms_norm(gpu_stream, &sequence_output, &normalization_weight, epsilon)?;
    // The precise float32 SwiGLU between the normalized recurrence output and
    // the output-gate projection, restoring the activation dtype afterwards.
    let float32_gate = ops::astype(gpu_stream, &output_gate, raw::mlx_dtype__MLX_FLOAT32)?;
    let gate_weights = ops::sigmoid(gpu_stream, &float32_gate)?;
    let activated_gate = ops::multiply(gpu_stream, &float32_gate, &gate_weights)?;
    let float32_output = ops::astype(gpu_stream, &normalized_output, raw::mlx_dtype__MLX_FLOAT32)?;
    let float32_gated = ops::multiply(gpu_stream, &activated_gate, &float32_output)?;
    let gated_output = ops::astype(gpu_stream, &float32_gated, output_gate.dtype().to_raw())?;
    let linear_value_dimension = linear_value_head_count * linear_head_dimension;
    let gated_output = ops::reshape(
        gpu_stream,
        &gated_output,
        &[1, row_count, linear_value_dimension],
    )?;
    let projected_output = trunk::quantized_matmul(
        gpu_stream,
        layer_quantization.output_projection,
        &gated_output,
        &output_projection,
    )?;
    let attention_residual = ops::add(gpu_stream, &hidden_states, &projected_output)?;
    let layer_output = trunk::trace_feed_forward_tail(
        gpu_stream,
        geometry,
        reader,
        layer_quantization.feed_forward_gate,
        layer_quantization.feed_forward_up,
        layer_quantization.feed_forward_down,
        attention_residual,
    )?;
    Ok((
        layer_output,
        next_rolling_state,
        current_recurrent_state,
        boundary_convolution_states,
        boundary_recurrent_states,
    ))
}
