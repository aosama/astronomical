//! Dual-output compiled composite for the gated-delta rate activations.
//!
//! The sigmoid over the update logits and the complete decay formula both
//! consume the two pre-activations of the fused rate projection, and each ran
//! as its own launch in every gated-delta layer of every decode step. One
//! multi-output compiled replay produces both, reading each input once. The
//! per-element arithmetic is the same primitive sequence the two separate
//! launches ran — sigmoid untouched, and the decay chain identical to
//! [`crate::compile::elementwise_graphs`]'s single-output graph — so the
//! replay matches the composed path bit for bit; the fusion parity contract
//! in the prism family's graphics-processor tests replays both routes over
//! adversarial inputs to prove it (issue #1366: decode is bound by launches
//! per token, and this removes one launch from every gated-delta layer).

use std::os::raw::c_int;

use crate::MlxCError;
use crate::compile::graph::{
    MlxCompiledGraph, array_from_vector, graph_output_array, set_graph_output_vector,
};
use crate::{MlxArray, MlxBindingsContext, MlxStream, raw};

const APPLY_GATED_DELTA_RATE_ACTIVATION_OPERATION: &str =
    "apply the compiled MLX gated-delta rate activation";
const COMPILE_GATED_DELTA_RATE_ACTIVATION_OPERATION: &str =
    "compile the shapeless MLX gated-delta rate activation graph";

/// Owns the compiled dual-output rate-activation graph.
///
/// Output order is fixed by the builder: the sigmoid update rates first, the
/// decay factors second. Callers consume them by that order.
#[derive(Debug)]
pub struct MlxCompiledGatedDeltaRateActivation {
    compiled_graph: MlxCompiledGraph,
}

impl MlxCompiledGatedDeltaRateActivation {
    /// Compiles the shapeless dual-output rate-activation graph.
    pub fn new() -> Result<Self, MlxCError> {
        Ok(Self {
            compiled_graph: MlxCompiledGraph::new(
                build_gated_delta_rate_activation_graph,
                COMPILE_GATED_DELTA_RATE_ACTIVATION_OPERATION,
                true,
            )?,
        })
    }
}

impl MlxBindingsContext {
    /// Applies the dual-output rate-activation graph, returning the update
    /// rates and the decay factors one replay produced.
    pub fn apply_compiled_gated_delta_rate_activation(
        &self,
        rate_activation: &MlxCompiledGatedDeltaRateActivation,
        update_logits: &MlxArray,
        decay_interval_inputs: &MlxArray,
        decay_rate_logarithm: &MlxArray,
        decay_interval_bias: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        let mut graph_outputs = rate_activation.compiled_graph.apply_multi(
            &[
                update_logits,
                decay_interval_inputs,
                decay_rate_logarithm,
                decay_interval_bias,
            ],
            APPLY_GATED_DELTA_RATE_ACTIVATION_OPERATION,
        )?;
        if graph_outputs.len() != 2 {
            return Err(MlxCError {
                operation: APPLY_GATED_DELTA_RATE_ACTIVATION_OPERATION,
                description: format!(
                    "compiled MLX gated-delta rate activation returned {} outputs, expected 2",
                    graph_outputs.len()
                ),
            });
        }
        let decay_rates = graph_outputs
            .pop()
            .expect("the length check proved two outputs");
        let update_rates = graph_outputs
            .pop()
            .expect("the length check proved two outputs");
        Ok((update_rates, decay_rates))
    }
}

unsafe extern "C" fn build_gated_delta_rate_activation_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 4 {
        return 1;
    }
    let update_logits = match array_from_vector(input_vector, 0) {
        Ok(update_logits) => update_logits,
        Err(get_status) => return get_status,
    };
    let decay_interval_inputs = match array_from_vector(input_vector, 1) {
        Ok(decay_interval_inputs) => decay_interval_inputs,
        Err(get_status) => return get_status,
    };
    let decay_rate_logarithm = match array_from_vector(input_vector, 2) {
        Ok(decay_rate_logarithm) => decay_rate_logarithm,
        Err(get_status) => return get_status,
    };
    let decay_interval_bias = match array_from_vector(input_vector, 3) {
        Ok(decay_interval_bias) => decay_interval_bias,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let update_rates = match graph_output_array(|output_array| {
        // SAFETY: The input and stream are live, and the output is uniquely writable.
        unsafe { raw::mlx_sigmoid(output_array, update_logits.raw(), gpu_stream.raw()) }
    }) {
        Ok(update_rates) => update_rates,
        Err(build_status) => return build_status,
    };
    // The decay chain below is the single-output gated-delta decay graph
    // verbatim: add bias, stable softplus through logaddexp(x, 0), the
    // float32 exponential of the per-head decay logarithm, and the closing
    // negate-and-exp product that keeps decay factors inside (0, 1).
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
    // Output order is the caller's ABI: update rates first, decay factors
    // second. Multi-output tracing only supports publishing the complete
    // result vector once at the end of the builder.
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output_vector(output_vector, &[&update_rates, &decays]) }
}
