use std::os::raw::c_int;

use crate::{
    mlx_compiled_graph::{array_from_vector, graph_output_array, set_graph_output},
    mlx_stream::MlxStream,
    raw,
};

pub(crate) unsafe extern "C" fn build_sparse_shared_expert_combination_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 3 {
        return 1;
    }
    let sparse_expert_output = match array_from_vector(input_vector, 0) {
        Ok(sparse_expert_output) => sparse_expert_output,
        Err(get_status) => return get_status,
    };
    let shared_expert_output = match array_from_vector(input_vector, 1) {
        Ok(shared_expert_output) => shared_expert_output,
        Err(get_status) => return get_status,
    };
    let shared_expert_gate_logits = match array_from_vector(input_vector, 2) {
        Ok(shared_expert_gate_logits) => shared_expert_gate_logits,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let shared_expert_gate_weights = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_sigmoid(
                output_array,
                shared_expert_gate_logits.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(shared_expert_gate_weights) => shared_expert_gate_weights,
        Err(build_status) => return build_status,
    };
    let gated_shared_expert_output = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                shared_expert_output.raw(),
                shared_expert_gate_weights.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(gated_shared_expert_output) => gated_shared_expert_output,
        Err(build_status) => return build_status,
    };
    let combined_expert_output = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_add(
                output_array,
                sparse_expert_output.raw(),
                gated_shared_expert_output.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(combined_expert_output) => combined_expert_output,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &combined_expert_output) }
}
