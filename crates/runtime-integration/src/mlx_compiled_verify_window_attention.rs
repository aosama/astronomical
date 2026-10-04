use super::VerifyWindowInputReader;
use super::trunk::{quantized_matmul, take_affine, trace_feed_forward_tail};
use crate::mlx_compiled_verify_window_geometry::VerifyWindowGeometry;
use crate::mlx_compiled_verify_window_ops as ops;
use crate::{MlxArray, MlxStream, raw};

pub(super) fn trace_full_attention_layer(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    reader: &mut VerifyWindowInputReader,
    input_vector: &raw::mlx_vector_array,
    layer_index: usize,
    hidden_states: MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray), i32> {
    let row_count = geometry.row_count();
    let query_head_count = geometry.query_head_count();
    let key_value_head_count = geometry.key_value_head_count();
    let head_dimension = geometry.attention_head_dimension();
    let rotary_dimension = geometry.rotary_dimension();
    let rope_base = geometry.rope_base();
    let epsilon = geometry.rms_norm_epsilon();

    let layer_quantization = geometry
        .layer_quantization(layer_index)
        .and_then(|layer_quantization| layer_quantization.full_attention)
        .ok_or(1)?;
    let keys_slab = reader.take()?;
    let values_slab = reader.take()?;
    let input_normalization_weight = reader.take()?;
    let query_projection = take_affine(reader)?;
    let key_projection = take_affine(reader)?;
    let value_projection = take_affine(reader)?;
    let output_projection = take_affine(reader)?;
    let query_normalization_weight = reader.take()?;
    let key_normalization_weight = reader.take()?;

    let position_offsets = ops::builder_input(*input_vector, 1)?;
    let key_value_base_offset = ops::builder_input(*input_vector, 2)?;
    // The key/value slab uses the attention layout [batch, kv_heads,
    // capacity, head_dim]; the capacity is the third axis.
    let slab_capacity = keys_slab.shape().get(2).copied().ok_or(1)?;

    let normalized_input = ops::fast_rms_norm(
        gpu_stream,
        &hidden_states,
        &input_normalization_weight,
        epsilon,
    )?;
    let query_and_gate = quantized_matmul(
        gpu_stream,
        layer_quantization.query,
        &normalized_input,
        &query_projection,
    )?;
    let query_and_gate = ops::reshape(
        gpu_stream,
        &query_and_gate,
        &[1, row_count, query_head_count, head_dimension * 2],
    )?;
    let queries = ops::slice(
        gpu_stream,
        &query_and_gate,
        &[0, 0, 0, 0],
        &[1, row_count, query_head_count, head_dimension],
        &[1, 1, 1, 1],
    )?;
    let output_gate = ops::slice(
        gpu_stream,
        &query_and_gate,
        &[0, 0, 0, head_dimension],
        &[1, row_count, query_head_count, head_dimension * 2],
        &[1, 1, 1, 1],
    )?;
    let output_gate = ops::reshape(
        gpu_stream,
        &output_gate,
        &[1, row_count, query_head_count * head_dimension],
    )?;
    let keys = quantized_matmul(
        gpu_stream,
        layer_quantization.key,
        &normalized_input,
        &key_projection,
    )?;
    let keys = ops::reshape(
        gpu_stream,
        &keys,
        &[1, row_count, key_value_head_count, head_dimension],
    )?;
    let values = quantized_matmul(
        gpu_stream,
        layer_quantization.value,
        &normalized_input,
        &value_projection,
    )?;
    let values = ops::reshape(
        gpu_stream,
        &values,
        &[1, row_count, key_value_head_count, head_dimension],
    )?;
    let normalized_queries =
        ops::fast_rms_norm(gpu_stream, &queries, &query_normalization_weight, epsilon)?;
    let normalized_keys =
        ops::fast_rms_norm(gpu_stream, &keys, &key_normalization_weight, epsilon)?;
    let transposed_queries = ops::transpose_axes(gpu_stream, &normalized_queries, &[0, 2, 1, 3])?;
    let transposed_keys = ops::transpose_axes(gpu_stream, &normalized_keys, &[0, 2, 1, 3])?;
    let transposed_values = ops::transpose_axes(gpu_stream, &values, &[0, 2, 1, 3])?;
    let rotated_queries = ops::rope_at_token_positions(
        gpu_stream,
        &transposed_queries,
        &position_offsets,
        rotary_dimension,
        rope_base,
    )?;
    let rotated_keys = ops::rope_at_token_positions(
        gpu_stream,
        &transposed_keys,
        &position_offsets,
        rotary_dimension,
        rope_base,
    )?;

    // The slabs hold the rotated prefix; garbage rows beyond the logical
    // offset are excluded by the additive per-row masks, and the new rotated
    // rows join through a static-shape concat.
    let prefix_positions = ops::arange_i32(gpu_stream, 0, slab_capacity)?;
    let window_positions = ops::arange_i32(gpu_stream, slab_capacity, row_count)?;
    let positions = ops::concatenate_axis(gpu_stream, &[&prefix_positions, &window_positions], 0)?;
    let extended_keys = ops::concatenate_axis(gpu_stream, &[&keys_slab, &rotated_keys], 2)?;
    let extended_values =
        ops::concatenate_axis(gpu_stream, &[&values_slab, &transposed_values], 2)?;
    let row_thresholds = ops::arange_i32(gpu_stream, 0, row_count)?;
    let mask_zero = ops::astype(
        gpu_stream,
        &MlxArray::from_f32(&[0.0], &[]).map_err(|_| 1)?,
        transposed_queries.dtype().to_raw(),
    )?;
    let mask_floor_value = ops::astype(
        gpu_stream,
        &MlxArray::from_f32(&[-1.0e30], &[]).map_err(|_| 1)?,
        transposed_queries.dtype().to_raw(),
    )?;
    let mut attention_rows = Vec::with_capacity(row_count as usize);
    for query_row_index in 0..row_count {
        let query_row = ops::slice(
            gpu_stream,
            &rotated_queries,
            &[0, 0, query_row_index, 0],
            &[1, query_head_count, query_row_index + 1, head_dimension],
            &[1, 1, 1, 1],
        )?;
        let row_threshold = ops::slice(
            gpu_stream,
            &row_thresholds,
            &[query_row_index],
            &[query_row_index + 1],
            &[1],
        )?;
        let visible_boundary = ops::add(gpu_stream, &key_value_base_offset, &row_threshold)?;
        let visible_positions = ops::greater_equal(gpu_stream, &visible_boundary, &positions)?;
        let additive_mask = ops::where_op(
            gpu_stream,
            &visible_positions,
            &mask_zero,
            &mask_floor_value,
        )?;
        attention_rows.push(ops::masked_row_attention(
            gpu_stream,
            &query_row,
            &extended_keys,
            &extended_values,
            geometry.attention_scale(),
            &additive_mask,
        )?);
    }
    let attention_row_references = attention_rows.iter().collect::<Vec<_>>();
    let attention_output = ops::concatenate_axis(gpu_stream, &attention_row_references, 2)?;
    let attention_output = ops::transpose_axes(gpu_stream, &attention_output, &[0, 2, 1, 3])?;
    let attention_output = ops::reshape(
        gpu_stream,
        &attention_output,
        &[1, row_count, query_head_count * head_dimension],
    )?;
    let gate_weights = ops::sigmoid(gpu_stream, &output_gate)?;
    let gated_output = ops::multiply(gpu_stream, &attention_output, &gate_weights)?;
    let projected_output = quantized_matmul(
        gpu_stream,
        layer_quantization.output,
        &gated_output,
        &output_projection,
    )?;
    let attention_residual = ops::add(gpu_stream, &hidden_states, &projected_output)?;
    let layer_output = trace_feed_forward_tail(
        gpu_stream,
        geometry,
        reader,
        layer_quantization.feed_forward_gate,
        layer_quantization.feed_forward_up,
        layer_quantization.feed_forward_down,
        attention_residual,
    )?;
    Ok((layer_output, rotated_keys, transposed_values))
}
