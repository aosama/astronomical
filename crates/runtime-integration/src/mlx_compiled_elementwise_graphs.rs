use std::os::raw::c_int;

use crate::mlx_compiled_attention_output_gate::build_attention_output_gate_graph;
use crate::mlx_compiled_sparse_shared_expert_combination::build_sparse_shared_expert_combination_graph;
use crate::{
    MlxArray, MlxRuntime, MlxRuntimeError,
    mlx_compiled_graph::{
        MlxCompiledGraph, array_from_vector, graph_output_array, set_graph_output,
    },
    mlx_stream::MlxStream,
    raw,
};

const APPLY_ATTENTION_OUTPUT_GATE_OPERATION: &str = "apply the compiled MLX attention output gate";
const COMPILE_ATTENTION_OUTPUT_GATE_OPERATION: &str =
    "compile the shapeless MLX attention output gate";
const APPLY_SPARSE_SHARED_EXPERT_COMBINATION_OPERATION: &str =
    "apply the compiled MLX sparse and shared expert combination";
const COMPILE_SPARSE_SHARED_EXPERT_COMBINATION_OPERATION: &str =
    "compile the shapeless MLX sparse and shared expert combination";
const APPLY_PRECISE_SWIGLU_OPERATION: &str = "apply the compiled precise MLX SwiGLU graph";
const COMPILE_PRECISE_SWIGLU_OPERATION: &str = "compile the shapeless precise MLX SwiGLU graph";
const APPLY_GATED_DELTA_DECAY_OPERATION: &str =
    "apply the compiled MLX gated-delta decay arithmetic";
const COMPILE_GATED_DELTA_DECAY_OPERATION: &str =
    "compile the shapeless MLX gated-delta decay arithmetic";

/// Retained shapeless compilations for elementwise Qwen3.5-MoE graph composites.
#[derive(Debug)]
pub struct MlxCompiledElementwiseGraphs {
    attention_output_gate: MlxCompiledGraph,
    gated_delta_decay: MlxCompiledGraph,
    precise_swiglu: MlxCompiledGraph,
    sparse_shared_expert_combination: MlxCompiledGraph,
}

impl MlxCompiledElementwiseGraphs {
    /// Creates reusable compiled graphs for elementwise model composites.
    pub fn new() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            attention_output_gate: MlxCompiledGraph::new(
                build_attention_output_gate_graph,
                COMPILE_ATTENTION_OUTPUT_GATE_OPERATION,
            )?,
            gated_delta_decay: MlxCompiledGraph::new(
                build_gated_delta_decay_graph,
                COMPILE_GATED_DELTA_DECAY_OPERATION,
            )?,
            precise_swiglu: MlxCompiledGraph::new(
                build_precise_swiglu_graph,
                COMPILE_PRECISE_SWIGLU_OPERATION,
            )?,
            sparse_shared_expert_combination: MlxCompiledGraph::new(
                build_sparse_shared_expert_combination_graph,
                COMPILE_SPARSE_SHARED_EXPERT_COMBINATION_OPERATION,
            )?,
        })
    }
}

impl MlxRuntime {
    /// Applies `attention_output * sigmoid(output_gate_logits)` as one compiled composite.
    pub fn apply_compiled_attention_output_gate(
        &self,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        attention_output: &MlxArray,
        output_gate_logits: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        compiled_elementwise_graphs.attention_output_gate.apply(
            &[attention_output, output_gate_logits],
            APPLY_ATTENTION_OUTPUT_GATE_OPERATION,
        )
    }

    /// Applies `sparse + shared * sigmoid(shared_gate_logits)` as one compiled composite.
    pub fn apply_compiled_sparse_shared_expert_combination(
        &self,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        sparse_expert_output: &MlxArray,
        shared_expert_output: &MlxArray,
        shared_expert_gate_logits: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        compiled_elementwise_graphs
            .sparse_shared_expert_combination
            .apply(
                &[
                    sparse_expert_output,
                    shared_expert_output,
                    shared_expert_gate_logits,
                ],
                APPLY_SPARSE_SHARED_EXPERT_COMBINATION_OPERATION,
            )
    }

    /// Applies float32 SwiGLU arithmetic and restores the input activation dtype.
    pub fn apply_compiled_precise_swiglu(
        &self,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        up_states: &MlxArray,
        gate_states: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        compiled_elementwise_graphs
            .precise_swiglu
            .apply(&[up_states, gate_states], APPLY_PRECISE_SWIGLU_OPERATION)
    }

    /// Applies the complete float32 gated-delta decay formula as one compiled composite.
    pub fn apply_compiled_gated_delta_decay(
        &self,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        decay_rate_logarithm: &MlxArray,
        decay_interval_inputs: &MlxArray,
        decay_interval_bias: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError> {
        compiled_elementwise_graphs.gated_delta_decay.apply(
            &[
                decay_rate_logarithm,
                decay_interval_inputs,
                decay_interval_bias,
            ],
            APPLY_GATED_DELTA_DECAY_OPERATION,
        )
    }
}

unsafe extern "C" fn build_precise_swiglu_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 2 {
        return 1;
    }
    let up_states = match array_from_vector(input_vector, 0) {
        Ok(up_states) => up_states,
        Err(get_status) => return get_status,
    };
    let gate_states = match array_from_vector(input_vector, 1) {
        Ok(gate_states) => gate_states,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let output_dtype = unsafe { raw::mlx_array_dtype(up_states.raw()) };
    let float32_gate = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                gate_states.raw(),
                raw::mlx_dtype__MLX_FLOAT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(float32_gate) => float32_gate,
        Err(build_status) => return build_status,
    };
    let gate_weights = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_sigmoid(output_array, float32_gate.raw(), gpu_stream.raw()) }
    }) {
        Ok(gate_weights) => gate_weights,
        Err(build_status) => return build_status,
    };
    let activated_gate = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                float32_gate.raw(),
                gate_weights.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(activated_gate) => activated_gate,
        Err(build_status) => return build_status,
    };
    let float32_up = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                up_states.raw(),
                raw::mlx_dtype__MLX_FLOAT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(float32_up) => float32_up,
        Err(build_status) => return build_status,
    };
    let float32_activated_states = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                activated_gate.raw(),
                float32_up.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(float32_activated_states) => float32_activated_states,
        Err(build_status) => return build_status,
    };
    let activated_states = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                float32_activated_states.raw(),
                output_dtype,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(activated_states) => activated_states,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &activated_states) }
}

unsafe extern "C" fn build_gated_delta_decay_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 3 {
        return 1;
    }
    let decay_rate_logarithm = match array_from_vector(input_vector, 0) {
        Ok(decay_rate_logarithm) => decay_rate_logarithm,
        Err(get_status) => return get_status,
    };
    let decay_interval_inputs = match array_from_vector(input_vector, 1) {
        Ok(decay_interval_inputs) => decay_interval_inputs,
        Err(get_status) => return get_status,
    };
    let decay_interval_bias = match array_from_vector(input_vector, 2) {
        Ok(decay_interval_bias) => decay_interval_bias,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let biased_decay_intervals = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_add(
                output_array,
                decay_interval_inputs.raw(),
                decay_interval_bias.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(biased_decay_intervals) => biased_decay_intervals,
        Err(build_status) => return build_status,
    };
    // Stable softplus through MLX logaddexp(x, 0).
    // The naive log1p(exp(x)) overflows to infinity for large positive decay
    // intervals, corrupting the gated-delta recurrent state during long-context
    // prefill. This matches MLX's nn.softplus, which delegates to logaddexp(x, 0).
    let biased_dtype = biased_decay_intervals.dtype();
    let scalar_shape_storage = [1_i32];
    let zero_decay_interval_scalar = match graph_output_array(|output_array| {
        // SAFETY: The shape pointer references live storage but the zero rank
        // means MLX reads no dimensions. Dtype and stream are valid, and the
        // output is uniquely writable. Broadcasting this scalar keeps the
        // compiled graph shapeless across variable token counts.
        unsafe {
            raw::mlx_zeros(
                output_array,
                scalar_shape_storage.as_ptr(),
                0,
                biased_dtype.to_raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(zero_decay_interval_scalar) => zero_decay_interval_scalar,
        Err(build_status) => return build_status,
    };
    let decay_intervals = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_logaddexp(
                output_array,
                biased_decay_intervals.raw(),
                zero_decay_interval_scalar.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decay_intervals) => decay_intervals,
        Err(build_status) => return build_status,
    };
    let float32_decay_logs = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_astype(
                output_array,
                decay_rate_logarithm.raw(),
                raw::mlx_dtype__MLX_FLOAT32,
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(float32_decay_logs) => float32_decay_logs,
        Err(build_status) => return build_status,
    };
    let decay_rates = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_exp(output_array, float32_decay_logs.raw(), gpu_stream.raw()) }
    }) {
        Ok(decay_rates) => decay_rates,
        Err(build_status) => return build_status,
    };
    let decay_products = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_multiply(
                output_array,
                decay_rates.raw(),
                decay_intervals.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decay_products) => decay_products,
        Err(build_status) => return build_status,
    };
    let negative_decay_products = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_negative(output_array, decay_products.raw(), gpu_stream.raw()) }
    }) {
        Ok(negative_decay_products) => negative_decay_products,
        Err(build_status) => return build_status,
    };
    let decays = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_exp(
                output_array,
                negative_decay_products.raw(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(decays) => decays,
        Err(build_status) => return build_status,
    };
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output(output_vector, &decays) }
}
