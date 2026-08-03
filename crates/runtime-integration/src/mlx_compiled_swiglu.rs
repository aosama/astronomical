use std::os::raw::c_int;

use crate::{
    MlxArray, MlxRuntime, MlxRuntimeError,
    mlx_compiled_graph::{
        MlxCompiledGraph, array_from_vector, graph_output_array, set_graph_output,
    },
    mlx_stream::MlxStream,
    raw,
};

const APPLY_COMPILED_SWIGLU_OPERATION: &str = "apply a compiled MLX SwiGLU graph";
const COMPILE_SWIGLU_OPERATION: &str = "compile a shapeless MLX SwiGLU graph";

/// Owned shapeless MLX compilation of `silu(gate) * input`.
#[derive(Debug)]
pub struct MlxCompiledSwiGlu {
    compiled_graph: MlxCompiledGraph,
}

impl MlxCompiledSwiGlu {
    /// Compiles one shape-polymorphic SwiGLU graph for reuse across model layers.
    pub fn new() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            compiled_graph: MlxCompiledGraph::new(build_swiglu_graph, COMPILE_SWIGLU_OPERATION)?,
        })
    }
}

impl MlxRuntime {
    /// Applies a retained shapeless SwiGLU compilation to broadcast-compatible inputs.
    pub fn apply_compiled_swiglu(
        &self,
        compiled_swiglu: &MlxCompiledSwiGlu,
        gate: &MlxArray,
        input: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        compiled_swiglu
            .compiled_graph
            .apply(&[gate, input], APPLY_COMPILED_SWIGLU_OPERATION)
    }
}

unsafe extern "C" fn build_swiglu_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() {
        return 1;
    }
    if unsafe { raw::mlx_vector_array_size(input_vector) } != 2 {
        return 1;
    }
    let gate = match array_from_vector(input_vector, 0) {
        Ok(gate) => gate,
        Err(status) => return status,
    };
    let input = match array_from_vector(input_vector, 1) {
        Ok(input) => input,
        Err(status) => return status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let sigmoid_gate = match graph_output_array(|output| {
        // SAFETY: Inputs and stream are live, and `output` is uniquely writable.
        unsafe { raw::mlx_sigmoid(output, gate.raw(), gpu_stream.raw()) }
    }) {
        Ok(sigmoid_gate) => sigmoid_gate,
        Err(status) => return status,
    };
    let activated_gate = match graph_output_array(|output| {
        // SAFETY: Inputs and stream are live, and `output` is uniquely writable.
        unsafe { raw::mlx_multiply(output, gate.raw(), sigmoid_gate.raw(), gpu_stream.raw()) }
    }) {
        Ok(activated_gate) => activated_gate,
        Err(status) => return status,
    };
    let swiglu_output = match graph_output_array(|output| {
        // SAFETY: Inputs and stream are live, and `output` is uniquely writable.
        unsafe { raw::mlx_multiply(output, activated_gate.raw(), input.raw(), gpu_stream.raw()) }
    }) {
        Ok(swiglu_output) => swiglu_output,
        Err(status) => return status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &swiglu_output) }
}
