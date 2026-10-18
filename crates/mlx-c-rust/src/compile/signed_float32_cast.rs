//! The shapeless compiled prologue of the Hadamard activation transform:
//! `astype(activations, float32) * signs`.
//!
//! Inputs:
//!   0. activations, any float dtype, last axis = the rotated width
//!   1. signs, float32 `[width]`, every value ±1
//!
//! Uncompiled, the cast and the multiply are two dispatches with a float32
//! intermediate round trip; MLX fuses both into one kernel under
//! compilation, with arithmetic identical to the two-op chain (the sign
//! multiply is exact in any float dtype).

use std::os::raw::c_int;

use crate::compile::graph::{array_from_vector, graph_output_array, set_graph_output};
use crate::{MlxStream, raw};

pub(crate) unsafe extern "C" fn build_signed_float32_cast_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 2 {
        return 1;
    }
    let activations = match array_from_vector(input_vector, 0) {
        Ok(activations) => activations,
        Err(get_status) => return get_status,
    };
    let signs = match array_from_vector(input_vector, 1) {
        Ok(signs) => signs,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let float32_activations = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                activations.raw(),
                raw::mlx_dtype__MLX_FLOAT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(float32_activations) => float32_activations,
        Err(build_status) => return build_status,
    };
    let signed_activations = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                float32_activations.raw(),
                signs.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(signed_activations) => signed_activations,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &signed_activations) }
}
