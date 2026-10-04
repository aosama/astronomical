//! Raw MLX operations for tracing the verification-window graph.
//!
//! Each helper mirrors the corresponding `MlxRuntime` method minus validation
//! and ownership ceremony: the builder runs inside an MLX compile trace where
//! the input ABI plan already validated every handle, panicking is forbidden,
//! and every failure path must surface as a plain status code. Helpers return
//! the owned lazy output so the builder chains them like the eager path does.

use crate::{MlxArray, MlxStream, array_from_vector, graph_output_array};
use astronomical_mlx_c_rust::raw;

pub(super) type BuildResult = Result<MlxArray, i32>;

pub(super) fn builder_input(
    input_vector: raw::mlx_vector_array,
    input_index: usize,
) -> Result<MlxArray, i32> {
    array_from_vector(input_vector, input_index)
}

pub(super) fn astype(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    dtype: raw::mlx_dtype,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_astype(output, input.raw(), dtype, gpu_stream.raw())
    })
}

pub(super) fn add(gpu_stream: &MlxStream, left: &MlxArray, right: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_add(output, left.raw(), right.raw(), gpu_stream.raw())
    })
}

pub(super) fn subtract(gpu_stream: &MlxStream, left: &MlxArray, right: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_subtract(output, left.raw(), right.raw(), gpu_stream.raw())
    })
}

pub(super) fn multiply(gpu_stream: &MlxStream, left: &MlxArray, right: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_multiply(output, left.raw(), right.raw(), gpu_stream.raw())
    })
}

pub(super) fn sigmoid(gpu_stream: &MlxStream, input: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe { raw::mlx_sigmoid(output, input.raw(), gpu_stream.raw()) })
}

pub(super) fn exponential(gpu_stream: &MlxStream, input: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe { raw::mlx_exp(output, input.raw(), gpu_stream.raw()) })
}

pub(super) fn negative(gpu_stream: &MlxStream, input: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe { raw::mlx_negative(output, input.raw(), gpu_stream.raw()) })
}

pub(super) fn logaddexp(gpu_stream: &MlxStream, left: &MlxArray, right: &MlxArray) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_logaddexp(output, left.raw(), right.raw(), gpu_stream.raw())
    })
}

pub(super) fn greater_equal(
    gpu_stream: &MlxStream,
    left: &MlxArray,
    right: &MlxArray,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_greater_equal(output, left.raw(), right.raw(), gpu_stream.raw())
    })
}

pub(super) fn where_op(
    gpu_stream: &MlxStream,
    condition: &MlxArray,
    taken_when_true: &MlxArray,
    taken_when_false: &MlxArray,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_where(
            output,
            condition.raw(),
            taken_when_true.raw(),
            taken_when_false.raw(),
            gpu_stream.raw(),
        )
    })
}

/// A scalar zero in the given dtype, mirroring the compiled decay graph's
/// stable-softplus identity operand.
pub(super) fn zero_scalar(gpu_stream: &MlxStream, dtype: raw::mlx_dtype) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_zeros(output, std::ptr::null(), 0, dtype, gpu_stream.raw())
    })
}

pub(super) fn take_axis_zero(
    gpu_stream: &MlxStream,
    source: &MlxArray,
    indices: &MlxArray,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_take_axis(output, source.raw(), indices.raw(), 0, gpu_stream.raw())
    })
}

pub(super) fn dequantize_affine(
    gpu_stream: &MlxStream,
    quantized_weights: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> BuildResult {
    let optional_group_size = raw::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let optional_bits = raw::mlx_optional_int {
        value: bits,
        has_value: true,
    };
    let absent_output_dtype = raw::mlx_optional_dtype {
        value: raw::mlx_dtype__MLX_FLOAT32,
        has_value: false,
    };
    graph_output_array(|output| unsafe {
        raw::mlx_dequantize(
            output,
            quantized_weights.raw(),
            scales.raw(),
            biases.raw(),
            optional_group_size,
            optional_bits,
            c"affine".as_ptr(),
            MlxArray::empty_raw(),
            absent_output_dtype,
            gpu_stream.raw(),
        )
    })
}

pub(super) fn quantized_matmul_affine(
    gpu_stream: &MlxStream,
    activations: &MlxArray,
    packed_weight: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    group_size: i32,
    bits: i32,
) -> BuildResult {
    let optional_group_size = raw::mlx_optional_int {
        value: group_size,
        has_value: true,
    };
    let optional_bits = raw::mlx_optional_int {
        value: bits,
        has_value: true,
    };
    graph_output_array(|output| unsafe {
        raw::mlx_quantized_matmul(
            output,
            activations.raw(),
            packed_weight.raw(),
            scales.raw(),
            biases.raw(),
            true,
            optional_group_size,
            optional_bits,
            c"affine".as_ptr(),
            gpu_stream.raw(),
        )
    })
}

pub(super) fn fast_rms_norm(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    weight: &MlxArray,
    epsilon: f32,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_fast_rms_norm(output, input.raw(), weight.raw(), epsilon, gpu_stream.raw())
    })
}

pub(super) fn reshape(gpu_stream: &MlxStream, input: &MlxArray, shape: &[i32]) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_reshape(
            output,
            input.raw(),
            shape.as_ptr(),
            shape.len(),
            gpu_stream.raw(),
        )
    })
}

pub(super) fn transpose_axes(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    axes: &[i32],
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_transpose_axes(
            output,
            input.raw(),
            axes.as_ptr(),
            axes.len(),
            gpu_stream.raw(),
        )
    })
}

pub(super) fn slice(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    start: &[i32],
    stop: &[i32],
    stride: &[i32],
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_slice(
            output,
            input.raw(),
            start.as_ptr(),
            start.len(),
            stop.as_ptr(),
            stop.len(),
            stride.as_ptr(),
            stride.len(),
            gpu_stream.raw(),
        )
    })
}

/// One owned raw vector of array handles for the vector-taking primitives.
fn raw_vector(parts: &[&MlxArray]) -> Result<raw::mlx_vector_array, i32> {
    let part_handles: Vec<raw::mlx_array> = parts.iter().map(|part| part.raw()).collect();
    let parts_vector =
        unsafe { raw::mlx_vector_array_new_data(part_handles.as_ptr(), part_handles.len()) };
    if parts_vector.ctx.is_null() {
        return Err(1);
    }
    Ok(parts_vector)
}

pub(super) fn concatenate_axis(
    gpu_stream: &MlxStream,
    parts: &[&MlxArray],
    axis: i32,
) -> BuildResult {
    let parts_vector = raw_vector(parts)?;
    let concatenate_status = graph_output_array(|output| unsafe {
        raw::mlx_concatenate_axis(output, parts_vector, axis, gpu_stream.raw())
    });
    // SAFETY: This local vector owner releases its live handle exactly once.
    unsafe { raw::mlx_vector_array_free(parts_vector) };
    concatenate_status
}

pub(super) fn stack_axis(gpu_stream: &MlxStream, parts: &[&MlxArray], axis: i32) -> BuildResult {
    let parts_vector = raw_vector(parts)?;
    let stack_status = graph_output_array(|output| unsafe {
        raw::mlx_stack_axis(output, parts_vector, axis, gpu_stream.raw())
    });
    // SAFETY: This local vector owner releases its live handle exactly once.
    unsafe { raw::mlx_vector_array_free(parts_vector) };
    stack_status
}

pub(super) fn squeeze_axis(gpu_stream: &MlxStream, input: &MlxArray, axis: i32) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_squeeze_axis(output, input.raw(), axis, gpu_stream.raw())
    })
}

pub(super) fn expand_dims(gpu_stream: &MlxStream, input: &MlxArray, axis: i32) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_expand_dims(output, input.raw(), axis, gpu_stream.raw())
    })
}

pub(super) fn repeat_axis(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    repetitions: i32,
    axis: i32,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_repeat_axis(output, input.raw(), repetitions, axis, gpu_stream.raw())
    })
}

pub(super) fn sum_axis_last(gpu_stream: &MlxStream, input: &MlxArray) -> BuildResult {
    let input_rank = input.shape().len() as i32;
    graph_output_array(|output| unsafe {
        raw::mlx_sum_axis(output, input.raw(), input_rank - 1, false, gpu_stream.raw())
    })
}

/// A static int32 ramp `[start, start + length)` — the position basis for
/// masked-attention masks and the per-row thresholds.
pub(super) fn arange_i32(gpu_stream: &MlxStream, start: i32, length: i32) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_arange(
            output,
            f64::from(start),
            f64::from(start) + f64::from(length),
            1.0,
            raw::mlx_dtype__MLX_INT32,
            gpu_stream.raw(),
        )
    })
}

pub(super) fn conv1d_depthwise(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    weight: &MlxArray,
    groups: i32,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_conv1d(
            output,
            input.raw(),
            weight.raw(),
            1,
            0,
            1,
            groups,
            gpu_stream.raw(),
        )
    })
}

/// Token-batched rotary embedding at per-row int32 positions, mirroring the
/// eager `rope_with_token_position_offsets` exactly (transpose in, dynamic
/// rope with generated frequencies and unit scale, transpose out).
pub(super) fn rope_at_token_positions(
    gpu_stream: &MlxStream,
    input: &MlxArray,
    token_position_offsets: &MlxArray,
    dimensions: i32,
    base: f32,
) -> BuildResult {
    let token_batched_input = transpose_axes(gpu_stream, input, &[2, 1, 0, 3])?;
    let optional_base = raw::mlx_optional_float {
        value: base,
        has_value: true,
    };
    let token_batched_output = graph_output_array(|output| unsafe {
        raw::mlx_fast_rope_dynamic(
            output,
            token_batched_input.raw(),
            dimensions,
            false,
            optional_base,
            1.0,
            token_position_offsets.raw(),
            MlxArray::empty_raw(),
            gpu_stream.raw(),
        )
    })?;
    transpose_axes(gpu_stream, &token_batched_output, &[2, 1, 0, 3])
}

/// Unmasked scaled dot-product attention for one query row, restricted by an
/// additive broadcast mask — the compiled mirror of the eager sequential
/// verification attention pass.
pub(super) fn masked_row_attention(
    gpu_stream: &MlxStream,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    scale: f32,
    additive_mask: &MlxArray,
) -> BuildResult {
    graph_output_array(|output| unsafe {
        raw::mlx_fast_scaled_dot_product_attention(
            output,
            queries.raw(),
            keys.raw(),
            values.raw(),
            scale,
            c"array".as_ptr(),
            additive_mask.raw(),
            MlxArray::empty_raw(),
            // mlx-c v0.7.0 exposes the fused-attention selection; the compiled
            // verification mirror keeps the reference implementation.
            false,
            gpu_stream.raw(),
        )
    })
}
