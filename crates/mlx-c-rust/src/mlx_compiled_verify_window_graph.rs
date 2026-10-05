//! The compiled MTP verification-window graph.
//!
//! The builder traces the whole hybrid decoder trunk for a fixed row count —
//! embedding lookup, every gated-delta and full-attention layer, the final
//! normalization, and the language-model head — as one positional closure over
//! the frozen [`crate::mlx_compiled_verify_window_geometry`] ABI. MLX compiles
//! that trace once per row count and replays it, eliminating the per-attempt
//! graph construction that dominated eager verification windows.
//!
//! Two structural rules, both measured or upstream-proven:
//!
//! - The MLX-C closure ABI has no capture context, so the geometry is served
//!   through a thread-local set around compile and every apply (an apply may
//!   re-trace when a state leaf's shape changes). This mirrors the upstream
//!   runner's trace-active flag.
//! - Graph builders publish outputs in ONE whole-vector set at the end:
//!   compiled-graph tracing can hand builders an output vector whose context
//!   is still null, which MLX's append operation rejects.
//!
//! The layer math deliberately mirrors the eager verification-window path op
//! for op — the prework convolution chain, the stable decay formula, the
//! float32 gated-delta recurrence, sequential masked attention rows, and the
//! precise SwiGLU activations — so the parity contract compares arithmetic,
//! not fused-kernel substitution.

use std::cell::Cell;

use crate::MlxBindingsContext;
use crate::mlx_compiled_verify_window_geometry::{
    VerifyWindowGeometry, VerifyWindowLayerKind, verify_window_input_slots,
};
use crate::mlx_compiled_verify_window_ops as ops;
use crate::{MlxArray, MlxStream, raw};
use crate::{MlxCError, MlxCompiledMultiOutputGraph, MlxMetalKernel, set_graph_output_vector};
#[path = "mlx_compiled_verify_window_attention.rs"]
mod attention;
#[path = "mlx_compiled_verify_window_gdn.rs"]
mod gdn;
#[path = "mlx_compiled_verify_window_trunk.rs"]
mod trunk;
#[path = "mlx_compiled_verify_window_trunk_tail.rs"]
mod trunk_tail;

const COMPILE_VERIFY_WINDOW_OPERATION: &str = "compile the MTP verification-window graph";
const APPLY_VERIFY_WINDOW_OPERATION: &str = "apply the MTP verification-window graph";

struct MlxCompiledVerifyWindowBuildContext<'a> {
    geometry: &'a VerifyWindowGeometry,
    gdn_kernels: VerifyWindowGdnKernelSet<'a>,
}

thread_local! {
    /// The borrowed trace context is pinned while MLX compiles or replays the
    /// closure, whose C ABI has no capture slot.
    static VERIFY_WINDOW_BUILD_CONTEXT: Cell<Option<*const ()>> =
        const { Cell::new(None) };
}

/// Fused GDN kernels and launch geometry supplied by the model-serving owner.
#[derive(Clone, Copy, Debug)]
pub struct VerifyWindowGdnKernelSet<'a> {
    pub prework_kernel: Option<&'a MlxMetalKernel>,
    pub checkpoint_kernel: Option<&'a MlxMetalKernel>,
    pub query_normalization_scale: &'a MlxArray,
    pub key_normalization_scale: &'a MlxArray,
    pub prework_lane_count: i32,
    pub checkpoint_threadgroup_thread_count: i32,
    pub checkpoint_value_row_block_size: i32,
}

fn with_build_context<R>(
    geometry: &VerifyWindowGeometry,
    gdn_kernels: VerifyWindowGdnKernelSet<'_>,
    trace_work: impl FnOnce() -> R,
) -> R {
    let context = MlxCompiledVerifyWindowBuildContext {
        geometry,
        gdn_kernels,
    };
    VERIFY_WINDOW_BUILD_CONTEXT.with(|cell| {
        cell.set(Some(
            (&context as *const MlxCompiledVerifyWindowBuildContext<'_>).cast(),
        ));
        let trace_result = trace_work();
        cell.set(None);
        trace_result
    })
}

fn current_build_context() -> Option<&'static MlxCompiledVerifyWindowBuildContext<'static>> {
    VERIFY_WINDOW_BUILD_CONTEXT.with(|cell| {
        cell.get().map(|context_pointer| unsafe {
            &*context_pointer.cast::<MlxCompiledVerifyWindowBuildContext<'static>>()
        })
    })
}

/// One row count's compiled verification window for one loaded model.
#[derive(Debug)]
pub struct MlxCompiledVerifyWindowGraph {
    compiled_graph: MlxCompiledMultiOutputGraph,
    geometry: VerifyWindowGeometry,
    input_slot_count: usize,
}

impl MlxCompiledVerifyWindowGraph {
    /// Compiles the window graph for one row count's geometry.
    pub fn new(
        geometry: VerifyWindowGeometry,
        gdn_kernels: VerifyWindowGdnKernelSet<'_>,
    ) -> Result<Self, MlxCError> {
        let input_slot_count = verify_window_input_slots(&geometry).len();
        let compiled_graph = with_build_context(&geometry, gdn_kernels, || {
            // Static shapes only: the row count is fixed per graph and the
            // key/value slab capacity is fixed per trace, so shapeless
            // compilation buys nothing and this MLX version's shapeless pass
            // cannot generalize static slices.
            MlxCompiledMultiOutputGraph::new(
                build_verify_window_graph,
                COMPILE_VERIFY_WINDOW_OPERATION,
                false,
            )
        })?;
        Ok(Self {
            compiled_graph,
            geometry,
            input_slot_count,
        })
    }

    /// The geometry this graph replays.
    pub fn geometry(&self) -> &VerifyWindowGeometry {
        &self.geometry
    }

    /// The frozen input-vector length this graph consumes.
    pub fn input_slot_count(&self) -> usize {
        self.input_slot_count
    }
}

impl MlxBindingsContext {
    /// Applies the compiled verification window and returns its ordered
    /// outputs. The geometry stays pinned for the call so a shape-change
    /// re-trace inside MLX can reach it.
    pub fn apply_compiled_verify_window_graph(
        &self,
        compiled_window: &MlxCompiledVerifyWindowGraph,
        graph_inputs: &[&MlxArray],
        gdn_kernels: VerifyWindowGdnKernelSet<'_>,
    ) -> Result<Vec<MlxArray>, MlxCError> {
        if graph_inputs.len() != compiled_window.input_slot_count {
            return Err(MlxCError {
                operation: APPLY_VERIFY_WINDOW_OPERATION,
                description: format!(
                    "the verification window expects {} inputs but received {}",
                    compiled_window.input_slot_count,
                    graph_inputs.len()
                ),
            });
        }
        with_build_context(&compiled_window.geometry, gdn_kernels, || {
            self.apply_compiled_multi_output_graph(&compiled_window.compiled_graph, graph_inputs)
        })
    }
}

unsafe extern "C" fn build_verify_window_graph(
    output_vector: *mut raw::mlx_vector_array,
    input_vector: raw::mlx_vector_array,
) -> std::os::raw::c_int {
    if output_vector.is_null() {
        return 1;
    }
    let Some(context) = current_build_context() else {
        return 1;
    };
    match trace_verify_window(context, input_vector) {
        Ok(window_outputs) => {
            let output_references = window_outputs.iter().collect::<Vec<_>>();
            // SAFETY: The output vector is unique and live for this publish.
            unsafe { set_graph_output_vector(output_vector, &output_references) }
        }
        Err(_status) => 1,
    }
}

/// Positional reader over the frozen input vector.
pub(super) struct VerifyWindowInputReader {
    input_vector: raw::mlx_vector_array,
    next_index: usize,
}

impl VerifyWindowInputReader {
    pub(super) fn take(&mut self) -> Result<MlxArray, i32> {
        let taken = ops::builder_input(self.input_vector, self.next_index)?;
        self.next_index += 1;
        Ok(taken)
    }

    /// Reads one array without consuming the sequential cursor — for the
    /// header and trunk slots that layer math needs at another moment.
    pub(super) fn take_at(&self, input_index: usize) -> Result<MlxArray, i32> {
        ops::builder_input(self.input_vector, input_index)
    }
}

#[allow(clippy::type_complexity)]
fn trace_verify_window(
    context: &MlxCompiledVerifyWindowBuildContext<'_>,
    input_vector: raw::mlx_vector_array,
) -> Result<Vec<MlxArray>, i32> {
    let geometry = context.geometry;
    let gpu_stream = MlxStream::default_gpu().map_err(|_| 1)?;
    let mut reader = VerifyWindowInputReader {
        input_vector,
        next_index: 5,
    };
    let mut layer_outputs = Vec::new();
    let (mut hidden_states, trunk_weight_indices) =
        trunk::trace_embedding_and_header(&gpu_stream, geometry, &input_vector)?;
    for layer_index in 0..geometry.layer_kinds().len() {
        let next_hidden_states = match geometry.layer_kinds()[layer_index] {
            VerifyWindowLayerKind::GatedDelta => {
                let (hidden, rolling, recurrent, boundary_convolutions, boundary_recurrents) =
                    gdn::trace_gated_delta_layer(
                        &gpu_stream,
                        geometry,
                        context.gdn_kernels,
                        &mut reader,
                        &input_vector,
                        layer_index,
                        hidden_states,
                    )?;
                layer_outputs.push(rolling);
                layer_outputs.push(recurrent);
                for (boundary_convolution, boundary_recurrent) in
                    boundary_convolutions.into_iter().zip(boundary_recurrents)
                {
                    layer_outputs.push(boundary_convolution);
                    layer_outputs.push(boundary_recurrent);
                }
                hidden
            }
            VerifyWindowLayerKind::FullAttention => {
                let (hidden, rotated_keys, values) = attention::trace_full_attention_layer(
                    &gpu_stream,
                    geometry,
                    &mut reader,
                    &input_vector,
                    layer_index,
                    hidden_states,
                )?;
                layer_outputs.push(rotated_keys);
                layer_outputs.push(values);
                hidden
            }
        };
        hidden_states = next_hidden_states;
    }
    let (logits, pre_final_hidden_states) = trunk_tail::trace_trunk_tail(
        &gpu_stream,
        geometry,
        &reader,
        &trunk_weight_indices,
        &hidden_states,
    )?;
    let mut window_outputs = vec![logits, pre_final_hidden_states];
    window_outputs.extend(layer_outputs);
    Ok(window_outputs)
}
