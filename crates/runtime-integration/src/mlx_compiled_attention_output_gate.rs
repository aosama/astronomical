use std::os::raw::c_int;

use crate::{
    mlx_compiled_graph::{array_from_vector, graph_output_array, set_graph_output},
    mlx_stream::MlxStream,
    raw,
};

pub(crate) unsafe extern "C" fn build_attention_output_gate_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 2 {
        return 1;
    }
    let attention_output = match array_from_vector(input_vector, 0) {
        Ok(attention_output) => attention_output,
        Err(get_status) => return get_status,
    };
    let output_gate_logits = match array_from_vector(input_vector, 1) {
        Ok(output_gate_logits) => output_gate_logits,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let output_gate_weights = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_sigmoid(output_array, output_gate_logits.raw(), gpu_stream.raw()) }
    }) {
        Ok(output_gate_weights) => output_gate_weights,
        Err(build_status) => return build_status,
    };
    let gated_attention_output = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                attention_output.raw(),
                output_gate_weights.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(gated_attention_output) => gated_attention_output,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &gated_attention_output) }
}
