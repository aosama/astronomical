//! Epsilon-specialized compiled composite fusing the gated-delta head
//! normalizations.
//!
//! The query/key head RMS normalizations of the gated-delta prework ran as
//! two launches per layer; one dual-output replay produces both. Every
//! primitive and its order match the composed path — `mlx_fast_rms_norm` is
//! the identical Metal kernel over its own input — and both the elementwise
//! fusion parity contract and the upstream greedy trajectory replay prove
//! the bits unchanged (issue #1366: decode is bound by launches per token).
//!
//! A second fusion was tried here and rejected by the trajectory replay:
//! chaining the output RMS normalization into the precise SwiGLU epilogue
//! passed synthetic bit parity at small widths yet diverged the real
//! upstream trajectory, because MLX's compiler reshapes the reduction
//! schedule when the elementwise epilogue is fused at the model's real
//! widths. Reduction-into-epilogue fusions stay composed until a kernel-
//! level contract can pin the summation order.
//!
//! Graph builders are plain function pointers that cannot capture the RMS
//! epsilon, and `mlx_fast_rms_norm` bakes its epsilon into the compiled
//! graph at trace time, so the epsilon rides in a process-wide bit slot for
//! the duration of each compilation. Model loads are sequential in a worker
//! process — one resident model at a time — which is the contract that keeps
//! the slot race-free; the compiled graphs keep their baked value forever
//! and later slot writes cannot affect them.

use std::os::raw::c_int;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::MlxCError;
use crate::compile::graph::{
    MlxCompiledGraph, array_from_vector, graph_output_array, set_graph_output_vector,
};
use crate::{MlxArray, MlxBindingsContext, MlxStream, raw};

static BAKED_RMS_NORM_EPSILON_BITS: AtomicU32 = AtomicU32::new(0);

fn baked_rms_norm_epsilon() -> f32 {
    f32::from_bits(BAKED_RMS_NORM_EPSILON_BITS.load(Ordering::Relaxed))
}

const APPLY_DUAL_HEAD_NORM_OPERATION: &str = "apply the compiled MLX dual head norm";
const COMPILE_DUAL_HEAD_NORM_OPERATION: &str = "compile the shapeless MLX dual head norm graph";

/// Owns the compiled dual-output query/key head-normalization graph.
///
/// Output order is fixed by the builder: the normalized queries first, the
/// normalized keys second. Callers consume them by that order.
#[derive(Debug)]
pub struct MlxCompiledDualHeadNorm {
    compiled_graph: MlxCompiledGraph,
}

impl MlxCompiledDualHeadNorm {
    /// Compiles the shapeless dual head-norm graph with the model's RMS
    /// epsilon baked in.
    pub fn new(rms_norm_epsilon: f32) -> Result<Self, MlxCError> {
        BAKED_RMS_NORM_EPSILON_BITS.store(rms_norm_epsilon.to_bits(), Ordering::Relaxed);
        Ok(Self {
            compiled_graph: MlxCompiledGraph::new(
                build_dual_head_norm_graph,
                COMPILE_DUAL_HEAD_NORM_OPERATION,
                true,
            )?,
        })
    }
}

impl MlxBindingsContext {
    /// Applies the dual head-norm graph, returning the normalized queries
    /// and keys one replay produced.
    pub fn apply_compiled_dual_head_norm(
        &self,
        dual_head_norm: &MlxCompiledDualHeadNorm,
        queries: &MlxArray,
        keys: &MlxArray,
        query_scale_weight: &MlxArray,
        key_scale_weight: &MlxArray,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        let mut graph_outputs = dual_head_norm.compiled_graph.apply_multi(
            &[queries, keys, query_scale_weight, key_scale_weight],
            APPLY_DUAL_HEAD_NORM_OPERATION,
        )?;
        if graph_outputs.len() != 2 {
            return Err(MlxCError {
                operation: APPLY_DUAL_HEAD_NORM_OPERATION,
                description: format!(
                    "compiled MLX dual head norm returned {} outputs, expected 2",
                    graph_outputs.len()
                ),
            });
        }
        let normalized_keys = graph_outputs
            .pop()
            .expect("the length check proved two outputs");
        let normalized_queries = graph_outputs
            .pop()
            .expect("the length check proved two outputs");
        Ok((normalized_queries, normalized_keys))
    }
}

unsafe extern "C" fn build_dual_head_norm_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> c_int {
    if output_vector.is_null() || unsafe { raw::mlx_vector_array_size(input_vector) } != 4 {
        return 1;
    }
    let queries = match array_from_vector(input_vector, 0) {
        Ok(queries) => queries,
        Err(get_status) => return get_status,
    };
    let keys = match array_from_vector(input_vector, 1) {
        Ok(keys) => keys,
        Err(get_status) => return get_status,
    };
    let query_scale_weight = match array_from_vector(input_vector, 2) {
        Ok(query_scale_weight) => query_scale_weight,
        Err(get_status) => return get_status,
    };
    let key_scale_weight = match array_from_vector(input_vector, 3) {
        Ok(key_scale_weight) => key_scale_weight,
        Err(get_status) => return get_status,
    };
    let gpu_stream = match MlxStream::default_gpu() {
        Ok(gpu_stream) => gpu_stream,
        Err(_) => return 1,
    };
    let normalized_queries = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_fast_rms_norm(
                output_array,
                queries.raw(),
                query_scale_weight.raw(),
                baked_rms_norm_epsilon(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(normalized_queries) => normalized_queries,
        Err(build_status) => return build_status,
    };
    let normalized_keys = match graph_output_array(|output_array| {
        // SAFETY: Inputs and stream are live, and the output is uniquely writable.
        unsafe {
            raw::mlx_fast_rms_norm(
                output_array,
                keys.raw(),
                key_scale_weight.raw(),
                baked_rms_norm_epsilon(),
                gpu_stream.raw(),
            )
        }
    }) {
        Ok(normalized_keys) => normalized_keys,
        Err(build_status) => return build_status,
    };
    // Output order is the caller's ABI: normalized queries first, keys
    // second. Multi-output tracing only supports publishing the complete
    // result vector once at the end of the builder.
    // SAFETY: The output vector is unique and live for this callback.
    unsafe { set_graph_output_vector(output_vector, &[&normalized_queries, &normalized_keys]) }
}
