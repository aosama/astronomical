//! Gated-delta recurrence tracing for the compiled verification window.
//!
//! Two interchangeable implementations behind one eligibility decision: the
//! fused path launches the model-owned prework and boundary-checkpoint Metal
//! kernels inside the trace, and the composed path mirrors the eager
//! verification window op for op. Both produce the same ordered result —
//! sequence output, next rolling state, next recurrent state, and per-row
//! boundary snapshots — so the layer tracer above them cannot tell them apart.

use crate::mlx_compiled_verify_window_geometry::VerifyWindowGeometry;
use crate::mlx_compiled_verify_window_graph::VerifyWindowGdnKernelSet;
use crate::mlx_compiled_verify_window_ops as ops;
use crate::{MlxMetalKernelOutput, MlxMetalKernelTemplateArgument, apply_metal_kernel_in_graph};
use astronomical_mlx_c_rust::{MlxArray, MlxDtype, MlxStream, raw};

/// The stable float32 decay formula and sigmoid update rates, mirroring the
/// compiled eager graphs op for op.
pub(super) fn trace_decays_and_rates(
    gpu_stream: &MlxStream,
    update_logits: &MlxArray,
    decay_inputs: &MlxArray,
    decay_interval_bias: &MlxArray,
    decay_rate_logarithm: &MlxArray,
) -> Result<(MlxArray, MlxArray), i32> {
    let update_rates = ops::sigmoid(gpu_stream, update_logits)?;
    let biased_intervals = ops::add(gpu_stream, decay_inputs, decay_interval_bias)?;
    let biased_dtype = biased_intervals.dtype().to_raw();
    let zero_interval = ops::zero_scalar(gpu_stream, biased_dtype)?;
    let decay_intervals = ops::logaddexp(gpu_stream, &biased_intervals, &zero_interval)?;
    let float32_decay_logs = ops::astype(
        gpu_stream,
        decay_rate_logarithm,
        raw::mlx_dtype__MLX_FLOAT32,
    )?;
    let decay_rates = ops::exponential(gpu_stream, &float32_decay_logs)?;
    let decay_products = ops::multiply(gpu_stream, &decay_rates, &decay_intervals)?;
    let negative_products = ops::negative(gpu_stream, &decay_products)?;
    let decays = ops::exponential(gpu_stream, &negative_products)?;
    Ok((update_rates, decays))
}

/// One float32 gated-delta recurrence step, mirroring
/// `qwen3_5_gated_delta_step` op for op.
fn traced_gated_delta_step(
    gpu_stream: &MlxStream,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    repeat_factor: i32,
    output_dtype: raw::mlx_dtype,
) -> Result<(MlxArray, MlxArray), i32> {
    let float32_queries = ops::astype(gpu_stream, queries, raw::mlx_dtype__MLX_FLOAT32)?;
    let float32_keys = ops::astype(gpu_stream, keys, raw::mlx_dtype__MLX_FLOAT32)?;
    let float32_values = ops::astype(gpu_stream, values, raw::mlx_dtype__MLX_FLOAT32)?;
    let float32_decays = ops::astype(gpu_stream, decays, raw::mlx_dtype__MLX_FLOAT32)?;
    let float32_update_rates = ops::astype(gpu_stream, update_rates, raw::mlx_dtype__MLX_FLOAT32)?;
    let repeated_queries = ops::repeat_axis(gpu_stream, &float32_queries, repeat_factor, 1)?;
    let repeated_keys = ops::repeat_axis(gpu_stream, &float32_keys, repeat_factor, 1)?;
    let decay_rows = ops::expand_dims(gpu_stream, &float32_decays, -1)?;
    let decay_matrices = ops::expand_dims(gpu_stream, &decay_rows, -1)?;
    let decayed_state = ops::multiply(gpu_stream, recurrent_state, &decay_matrices)?;
    let expanded_keys = ops::expand_dims(gpu_stream, &repeated_keys, 2)?;
    let state_key_products = ops::multiply(gpu_stream, &decayed_state, &expanded_keys)?;
    let remembered_values = ops::sum_axis_last(gpu_stream, &state_key_products)?;
    let value_differences = ops::subtract(gpu_stream, &float32_values, &remembered_values)?;
    let expanded_update_rates = ops::expand_dims(gpu_stream, &float32_update_rates, -1)?;
    let value_updates = ops::multiply(gpu_stream, &value_differences, &expanded_update_rates)?;
    let expanded_value_updates = ops::expand_dims(gpu_stream, &value_updates, -1)?;
    let state_updates = ops::multiply(gpu_stream, &expanded_keys, &expanded_value_updates)?;
    let next_recurrent_state = ops::add(gpu_stream, &decayed_state, &state_updates)?;
    let expanded_queries = ops::expand_dims(gpu_stream, &repeated_queries, 2)?;
    let state_query_products = ops::multiply(gpu_stream, &next_recurrent_state, &expanded_queries)?;
    let float32_output = ops::sum_axis_last(gpu_stream, &state_query_products)?;
    let output = ops::astype(gpu_stream, &float32_output, output_dtype)?;
    Ok((output, next_recurrent_state))
}

#[allow(clippy::type_complexity)]
pub(super) fn trace_fused_gated_delta(
    gpu_stream: &MlxStream,
    kernels: VerifyWindowGdnKernelSet<'_>,
    geometry: &VerifyWindowGeometry,
    rolling_state: &MlxArray,
    convolution_input: &MlxArray,
    recurrent_state: &MlxArray,
    mixed_queries_keys_values: &MlxArray,
    convolution_weight: &MlxArray,
    query_normalization_scale: &MlxArray,
    key_normalization_scale: &MlxArray,
    update_rates: &MlxArray,
    decays: &MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray, Vec<MlxArray>, Vec<MlxArray>), i32> {
    let row_count = geometry.row_count();
    let key_head_count = geometry.linear_key_head_count();
    let value_head_count = geometry.linear_value_head_count();
    let head_dimension = geometry.linear_head_dimension();
    let convolution_dimension = geometry.linear_convolution_dimension();
    let rolling_row_count = geometry.linear_convolution_kernel_dimension() - 1;
    let prework_kernel = kernels.prework_kernel.ok_or(1)?;
    let prework_outputs = apply_metal_kernel_in_graph(
        prework_kernel,
        &[
            mixed_queries_keys_values,
            rolling_state,
            convolution_weight,
            query_normalization_scale,
            key_normalization_scale,
        ],
        &[
            MlxMetalKernelOutput::new(
                vec![1, row_count, key_head_count, head_dimension],
                MlxDtype::BFloat16,
            ),
            MlxMetalKernelOutput::new(
                vec![1, row_count, key_head_count, head_dimension],
                MlxDtype::BFloat16,
            ),
            MlxMetalKernelOutput::new(
                vec![1, row_count, value_head_count, head_dimension],
                MlxDtype::BFloat16,
            ),
            MlxMetalKernelOutput::new(
                vec![1, rolling_row_count, convolution_dimension],
                MlxDtype::BFloat16,
            ),
        ],
        [
            kernels.prework_lane_count,
            row_count,
            2 * key_head_count + value_head_count,
        ],
        [kernels.prework_lane_count, 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "T",
                dtype: MlxDtype::BFloat16,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "HK",
                integer_template_argument: key_head_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "HV",
                integer_template_argument: value_head_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "DK",
                integer_template_argument: head_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "DV",
                integer_template_argument: head_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "NKEEP",
                integer_template_argument: rolling_row_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "C",
                integer_template_argument: convolution_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "S",
                integer_template_argument: row_count,
            },
        ],
        gpu_stream,
    )?;
    let mut prework_outputs = prework_outputs.into_iter();
    let queries = prework_outputs.next().ok_or(1)?;
    let keys = prework_outputs.next().ok_or(1)?;
    let values = prework_outputs.next().ok_or(1)?;
    let next_rolling_state = prework_outputs.next().ok_or(1)?;

    let token_count_input = MlxArray::from_i32(&[row_count], &[]).map_err(|_| 1)?;
    let first_checkpoint_input = MlxArray::from_i32(&[1], &[]).map_err(|_| 1)?;
    let checkpoint_interval_input = MlxArray::from_i32(&[1], &[]).map_err(|_| 1)?;
    let checkpoint_count = row_count - 1;
    let checkpoint_count_input = MlxArray::from_i32(&[checkpoint_count], &[]).map_err(|_| 1)?;
    let checkpoint_kernel = kernels.checkpoint_kernel.ok_or(1)?;
    let checkpoint_outputs = apply_metal_kernel_in_graph(
        checkpoint_kernel,
        &[
            &queries,
            &keys,
            &values,
            decays,
            update_rates,
            recurrent_state,
            &token_count_input,
            &first_checkpoint_input,
            &checkpoint_interval_input,
            &checkpoint_count_input,
        ],
        &[
            MlxMetalKernelOutput::new(
                vec![1, row_count, value_head_count, head_dimension],
                MlxDtype::BFloat16,
            ),
            MlxMetalKernelOutput::new(recurrent_state.shape(), MlxDtype::Float32),
            MlxMetalKernelOutput::new(
                vec![
                    checkpoint_count,
                    1,
                    value_head_count,
                    head_dimension,
                    head_dimension,
                ],
                MlxDtype::Float32,
            ),
        ],
        [
            kernels.checkpoint_threadgroup_thread_count
                * (head_dimension / kernels.checkpoint_value_row_block_size),
            value_head_count,
            1,
        ],
        [kernels.checkpoint_threadgroup_thread_count, 1, 1],
        &[
            MlxMetalKernelTemplateArgument::Dtype {
                name: "InT",
                dtype: MlxDtype::BFloat16,
            },
            MlxMetalKernelTemplateArgument::Dtype {
                name: "StT",
                dtype: MlxDtype::Float32,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "Dk",
                integer_template_argument: head_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "Dv",
                integer_template_argument: head_dimension,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "Hk",
                integer_template_argument: key_head_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "Hv",
                integer_template_argument: value_head_count,
            },
            MlxMetalKernelTemplateArgument::Integer {
                name: "B",
                integer_template_argument: 1,
            },
        ],
        gpu_stream,
    )?;
    let mut checkpoint_outputs = checkpoint_outputs.into_iter();
    let sequence_output = checkpoint_outputs.next().ok_or(1)?;
    let next_recurrent_state = checkpoint_outputs.next().ok_or(1)?;
    let packed_boundary_states = checkpoint_outputs.next().ok_or(1)?;
    let recurrent_state_shape = recurrent_state.shape();
    let mut boundary_recurrent_states = Vec::with_capacity(checkpoint_count as usize);
    for checkpoint_index in 0..checkpoint_count {
        let sliced_boundary = ops::slice(
            gpu_stream,
            &packed_boundary_states,
            &[checkpoint_index, 0, 0, 0, 0],
            &[
                checkpoint_index + 1,
                1,
                value_head_count,
                head_dimension,
                head_dimension,
            ],
            &[1, 1, 1, 1, 1],
        )?;
        boundary_recurrent_states.push(ops::reshape(
            gpu_stream,
            &sliced_boundary,
            &recurrent_state_shape,
        )?);
    }
    let mut boundary_convolution_states = Vec::with_capacity(checkpoint_count as usize);
    for consumed_rows in 1..row_count {
        boundary_convolution_states.push(ops::slice(
            gpu_stream,
            convolution_input,
            &[0, consumed_rows, 0],
            &[1, consumed_rows + rolling_row_count, convolution_dimension],
            &[1, 1, 1],
        )?);
    }
    Ok((
        sequence_output,
        next_rolling_state,
        next_recurrent_state,
        boundary_convolution_states,
        boundary_recurrent_states,
    ))
}

#[allow(clippy::type_complexity)]
pub(super) fn trace_composed_gated_delta(
    gpu_stream: &MlxStream,
    geometry: &VerifyWindowGeometry,
    convolution_input: &MlxArray,
    recurrent_state: &MlxArray,
    convolution_weight: &MlxArray,
    query_normalization_scale: &MlxArray,
    key_normalization_scale: &MlxArray,
    update_rates: &MlxArray,
    decays: &MlxArray,
) -> Result<(MlxArray, MlxArray, MlxArray, Vec<MlxArray>, Vec<MlxArray>), i32> {
    let row_count = geometry.row_count();
    let linear_head_dimension = geometry.linear_head_dimension();
    let linear_key_dimension = geometry.linear_key_dimension();
    let linear_convolution_dimension = geometry.linear_convolution_dimension();
    let linear_key_head_count = geometry.linear_key_head_count();
    let linear_value_head_count = geometry.linear_value_head_count();
    let epsilon = geometry.rms_norm_epsilon();
    let rolling_row_count = geometry.linear_convolution_kernel_dimension() - 1;
    let convolution_length = rolling_row_count + row_count;
    let next_rolling_state = ops::slice(
        gpu_stream,
        convolution_input,
        &[0, convolution_length - rolling_row_count, 0],
        &[1, convolution_length, linear_convolution_dimension],
        &[1, 1, 1],
    )?;
    let convolution_output = ops::conv1d_depthwise(
        gpu_stream,
        convolution_input,
        convolution_weight,
        linear_convolution_dimension,
    )?;
    let sigmoid_output = ops::sigmoid(gpu_stream, &convolution_output)?;
    let activated_output = ops::multiply(gpu_stream, &convolution_output, &sigmoid_output)?;
    let queries = ops::slice(
        gpu_stream,
        &activated_output,
        &[0, 0, 0],
        &[1, row_count, linear_key_dimension],
        &[1, 1, 1],
    )?;
    let queries = ops::reshape(
        gpu_stream,
        &queries,
        &[1, row_count, linear_key_head_count, linear_head_dimension],
    )?;
    let keys = ops::slice(
        gpu_stream,
        &activated_output,
        &[0, 0, linear_key_dimension],
        &[1, row_count, linear_key_dimension * 2],
        &[1, 1, 1],
    )?;
    let keys = ops::reshape(
        gpu_stream,
        &keys,
        &[1, row_count, linear_key_head_count, linear_head_dimension],
    )?;
    let values = ops::slice(
        gpu_stream,
        &activated_output,
        &[0, 0, linear_key_dimension * 2],
        &[1, row_count, linear_convolution_dimension],
        &[1, 1, 1],
    )?;
    let values = ops::reshape(
        gpu_stream,
        &values,
        &[1, row_count, linear_value_head_count, linear_head_dimension],
    )?;
    let queries = ops::fast_rms_norm(gpu_stream, &queries, query_normalization_scale, epsilon)?;
    let keys = ops::fast_rms_norm(gpu_stream, &keys, key_normalization_scale, epsilon)?;
    let float32_recurrent = ops::astype(gpu_stream, recurrent_state, raw::mlx_dtype__MLX_FLOAT32)?;
    let repeat_factor = linear_value_head_count / linear_key_head_count;
    let activation_dtype = queries.dtype().to_raw();
    let mut token_outputs = Vec::with_capacity(row_count as usize);
    let mut boundary_convolution_states = Vec::with_capacity(row_count.saturating_sub(1) as usize);
    let mut boundary_recurrent_states = Vec::with_capacity(row_count.saturating_sub(1) as usize);
    let mut current_recurrent_state = float32_recurrent;
    for token_index in 0..row_count {
        let token_queries = sliced_token_rank_four(
            gpu_stream,
            &queries,
            token_index,
            linear_key_head_count,
            linear_head_dimension,
        )?;
        let token_keys = sliced_token_rank_four(
            gpu_stream,
            &keys,
            token_index,
            linear_key_head_count,
            linear_head_dimension,
        )?;
        let token_values = sliced_token_rank_four(
            gpu_stream,
            &values,
            token_index,
            linear_value_head_count,
            linear_head_dimension,
        )?;
        let token_decays =
            sliced_token_rank_three(gpu_stream, decays, token_index, linear_value_head_count)?;
        let token_update_rates = sliced_token_rank_three(
            gpu_stream,
            update_rates,
            token_index,
            linear_value_head_count,
        )?;
        let (token_output, next_recurrent_state) = traced_gated_delta_step(
            gpu_stream,
            &token_queries,
            &token_keys,
            &token_values,
            &token_decays,
            &token_update_rates,
            &current_recurrent_state,
            repeat_factor,
            activation_dtype,
        )?;
        if token_index + 1 < row_count {
            let consumed_rows = token_index + 1;
            boundary_convolution_states.push(ops::slice(
                gpu_stream,
                convolution_input,
                &[0, consumed_rows, 0],
                &[
                    1,
                    consumed_rows + rolling_row_count,
                    linear_convolution_dimension,
                ],
                &[1, 1, 1],
            )?);
            boundary_recurrent_states.push(current_recurrent_state.retain().map_err(|_| 1)?);
        }
        token_outputs.push(token_output);
        current_recurrent_state = next_recurrent_state;
    }
    let sequence_output_references = token_outputs.iter().collect::<Vec<_>>();
    let sequence_output = ops::stack_axis(gpu_stream, &sequence_output_references, 1)?;
    Ok((
        sequence_output,
        next_rolling_state,
        current_recurrent_state,
        boundary_convolution_states,
        boundary_recurrent_states,
    ))
}

fn sliced_token_rank_four(
    gpu_stream: &MlxStream,
    sequence: &MlxArray,
    token_index: i32,
    head_count: i32,
    head_dimension: i32,
) -> Result<MlxArray, i32> {
    let sliced_token = ops::slice(
        gpu_stream,
        sequence,
        &[0, token_index, 0, 0],
        &[1, token_index + 1, head_count, head_dimension],
        &[1, 1, 1, 1],
    )?;
    ops::squeeze_axis(gpu_stream, &sliced_token, 1)
}

fn sliced_token_rank_three(
    gpu_stream: &MlxStream,
    sequence: &MlxArray,
    token_index: i32,
    head_count: i32,
) -> Result<MlxArray, i32> {
    let sliced_token = ops::slice(
        gpu_stream,
        sequence,
        &[0, token_index, 0],
        &[1, token_index + 1, head_count],
        &[1, 1, 1],
    )?;
    ops::squeeze_axis(gpu_stream, &sliced_token, 1)
}
