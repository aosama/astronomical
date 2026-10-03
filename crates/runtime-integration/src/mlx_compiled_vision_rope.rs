use std::os::raw::c_int;

use crate::{
    mlx_array_vector::MlxArrayVector,
    mlx_compiled_graph::{array_from_vector, graph_output_array, set_graph_output},
    mlx_stream::MlxStream,
    raw,
};

/// Builds the shapeless compiled graph for Qwen3-VL rotate-half rotary embedding.
///
/// Inputs:
///   0. attention states `[patch, head, head_dim]`
///   1. rotary cosines   `[patch, 1, head_dim]` float32
///   2. rotary sines     `[patch, 1, head_dim]` float32
///   3. first half       `[patch, head, head_dim / 2]` (slice of input 0)
///   4. second half      `[patch, head, head_dim / 2]` (slice of input 0)
///
/// The graph computes `astype(add(multiply(x, cos), multiply(rotated, sin)), x.dtype)`
/// where `rotated = concat(negative(second_half), first_half, axis=2)`.
///
/// The two head-half slices are deliberately kept OUTSIDE this compiled graph.
/// The pinned MLX (0.32.3) `Slice` primitive has no `output_shapes` override, so
/// shapeless compilation cannot re-derive a slice's output shape for the varying
/// patch count; the graph therefore receives the pre-sliced halves and only fuses
/// the elementwise tail (the concat remains one shape kernel).
pub(crate) unsafe extern "C" fn build_vision_rope_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 5 {
        return 1;
    }
    let attention_states = match array_from_vector(input_vector, 0) {
        Ok(attention_states) => attention_states,
        Err(get_status) => return get_status,
    };
    let rotary_cosines = match array_from_vector(input_vector, 1) {
        Ok(rotary_cosines) => rotary_cosines,
        Err(get_status) => return get_status,
    };
    let rotary_sines = match array_from_vector(input_vector, 2) {
        Ok(rotary_sines) => rotary_sines,
        Err(get_status) => return get_status,
    };
    let first_half = match array_from_vector(input_vector, 3) {
        Ok(first_half) => first_half,
        Err(get_status) => return get_status,
    };
    let second_half = match array_from_vector(input_vector, 4) {
        Ok(second_half) => second_half,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let output_dtype = unsafe { raw::mlx_array_dtype(attention_states.raw()) };
    let negative_second_half = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_negative(output_array, second_half.raw(), gpu_stream.raw()) }
    }) {
        Ok(negative_second_half) => negative_second_half,
        Err(build_status) => return build_status,
    };
    let rotated_inputs = match MlxArrayVector::new(&[&negative_second_half, &first_half]) {
        Ok(rotated_inputs) => rotated_inputs,
        Err(_) => return 1,
    };
    let rotated_states = match graph_output_array(|output_array| {
        // SAFETY: The vector and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_concatenate_axis(output_array, rotated_inputs.raw(), 2, gpu_stream.raw())
        }
    }) {
        Ok(rotated_states) => rotated_states,
        Err(build_status) => return build_status,
    };
    let cosine_component = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                attention_states.raw(),
                rotary_cosines.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(cosine_component) => cosine_component,
        Err(build_status) => return build_status,
    };
    let sine_component = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                rotated_states.raw(),
                rotary_sines.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(sine_component) => sine_component,
        Err(build_status) => return build_status,
    };
    let summed_components = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_add(
                output_array,
                cosine_component.raw(),
                sine_component.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(summed_components) => summed_components,
        Err(build_status) => return build_status,
    };
    let rotated_states_output = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                summed_components.raw(),
                output_dtype,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(rotated_states_output) => rotated_states_output,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &rotated_states_output) }
}
