use std::os::raw::c_int;

use crate::{
    MlxArray, MlxRuntimeError, mlx_array_vector::MlxArrayVector, mlx_runtime::check_status,
};
use astronomical_mlx_c_rust::raw;

/// The MLX C closure ABI every compiled-graph builder follows: it receives the
/// output vector to populate and the input vector to read, and returns zero on
/// success.
pub type MlxGraphBuilder =
    unsafe extern "C" fn(*mut raw::mlx_vector_array, raw::mlx_vector_array) -> c_int;

/// Owns one compiled MLX graph with a single array output.
///
/// The closure ABI already returns an output vector, so one compiled graph
/// serves both single-output [`MlxCompiledGraph::apply`] and multi-output
/// [`MlxCompiledGraph::apply_multi`] replays.
#[derive(Debug)]
pub(crate) struct MlxCompiledGraph {
    compiled_closure: MlxClosure,
}

impl MlxCompiledGraph {
    /// Compiles one graph. `shapeless` selects MLX's shape-polymorphic
    /// tracing: the model's shapeless elementwise graphs need it, but it
    /// rejects static slices on this MLX version (their output shape cannot
    /// be generalized), so static-shape graphs like the verification window
    /// compile with `false`.
    pub(crate) fn new(
        graph_builder: MlxGraphBuilder,
        compile_operation: &'static str,
        shapeless: bool,
    ) -> Result<Self, MlxRuntimeError> {
        let source_closure = MlxClosure::from_function(graph_builder, compile_operation)?;
        let mut compiled_closure = MlxClosure::empty();
        // SAFETY: Both closure handles are live and uniquely owned. MLX copies
        // the compiled function into `compiled_closure`.
        let compile_status = unsafe {
            raw::mlx_compile(compiled_closure.raw_mut(), source_closure.raw(), shapeless)
        };
        check_status(compile_status, compile_operation)?;
        compiled_closure.require_populated(compile_operation)?;
        Ok(Self { compiled_closure })
    }

    pub(crate) fn apply(
        &self,
        graph_inputs: &[&MlxArray],
        apply_operation: &'static str,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let input_vector = MlxArrayVector::new(graph_inputs)?;
        let mut output_vector = MlxArrayVector::empty(apply_operation)?;
        // SAFETY: The compiled closure and input vector remain live for the
        // synchronous graph-building call, and the output vector is unique.
        let apply_status = unsafe {
            raw::mlx_closure_apply(
                output_vector.raw_mut(),
                self.compiled_closure.raw(),
                input_vector.raw(),
            )
        };
        check_status(apply_status, apply_operation)?;
        if output_vector.len() != 1 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: apply_operation,
                description: format!(
                    "compiled MLX graph returned {} outputs instead of one",
                    output_vector.len()
                ),
            });
        }
        output_vector.array_at(0, apply_operation)
    }

    /// Applies the compiled graph and returns every output it produced.
    ///
    /// Multi-output graphs order their outputs by position, exactly like the
    /// graph builder wrote them; callers must consume them by that fixed ABI.
    pub(crate) fn apply_multi(
        &self,
        graph_inputs: &[&MlxArray],
        apply_operation: &'static str,
    ) -> Result<Vec<MlxArray>, MlxRuntimeError> {
        let input_vector = MlxArrayVector::new(graph_inputs)?;
        let mut output_vector = MlxArrayVector::empty(apply_operation)?;
        // SAFETY: The compiled closure and input vector remain live for the
        // synchronous graph-building call, and the output vector is unique.
        let apply_status = unsafe {
            raw::mlx_closure_apply(
                output_vector.raw_mut(),
                self.compiled_closure.raw(),
                input_vector.raw(),
            )
        };
        check_status(apply_status, apply_operation)?;
        let output_count = output_vector.len();
        if output_count == 0 {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: apply_operation,
                description: "compiled MLX graph returned no outputs".to_owned(),
            });
        }
        (0..output_count)
            .map(|output_index| output_vector.array_at(output_index, apply_operation))
            .collect()
    }
}

/// Reads one array out of a closure's input vector inside a graph builder.
pub fn array_from_vector(
    input_vector: raw::mlx_vector_array,
    input_index: usize,
) -> Result<MlxArray, c_int> {
    let mut input_array = MlxArray::empty();
    // SAFETY: The input vector is live for the callback and the destination
    // array owner is uniquely writable. MLX copies the selected handle.
    let get_status =
        unsafe { raw::mlx_vector_array_get(input_array.raw_mut(), input_vector, input_index) };
    if get_status != 0 || input_array.is_empty() {
        return Err(if get_status == 0 { 1 } else { get_status });
    }
    Ok(input_array)
}

/// Runs one raw MLX operation inside a graph builder and owns its lazy output.
pub fn graph_output_array(
    build_graph: impl FnOnce(*mut raw::mlx_array) -> c_int,
) -> Result<MlxArray, c_int> {
    let mut output_array = MlxArray::empty();
    let build_status = build_graph(output_array.raw_mut());
    if build_status != 0 || output_array.is_empty() {
        return Err(if build_status == 0 { 1 } else { build_status });
    }
    Ok(output_array)
}

/// Publishes the single output of a single-output graph builder.
pub unsafe fn set_graph_output(
    output_vector: *mut raw::mlx_vector_array,
    graph_output: &MlxArray,
) -> c_int {
    // SAFETY: The caller guarantees that the destination vector is unique and
    // live. MLX copies the lazy output handle before local owners are released.
    unsafe { raw::mlx_vector_array_set_value(output_vector, graph_output.raw()) }
}

/// Appends one lazy output to a multi-output graph's result vector.
///
/// MLX appends through the vector's context pointer, so pass-by-value still
/// mutates the caller's vector; the pointer is dereferenced once for the copy.
/// Publishes a multi-output graph's complete ordered result vector at once.
///
/// Compiled-graph tracing hands builders an output vector whose context may
/// still be null, and MLX's append operation throws on that null context while
/// its whole-vector set allocates on demand; collecting the outputs locally and
/// publishing them once at the end of the builder is therefore the only trace-
/// safe multi-output pattern.
pub unsafe fn set_graph_output_vector(
    output_vector: *mut raw::mlx_vector_array,
    graph_outputs: &[&MlxArray],
) -> c_int {
    let output_handles: Vec<raw::mlx_array> = graph_outputs.iter().map(|o| o.raw()).collect();
    // SAFETY: The caller guarantees that the destination vector is unique and
    // live. MLX copies the lazy output handles before local owners are released.
    unsafe {
        raw::mlx_vector_array_set_data(output_vector, output_handles.as_ptr(), output_handles.len())
    }
}

#[derive(Debug)]
struct MlxClosure {
    raw_closure: raw::mlx_closure,
}

impl MlxClosure {
    fn empty() -> Self {
        // SAFETY: MLX returns its documented null-context output placeholder;
        // `mlx_compile` must populate it before callers can apply the closure.
        let raw_closure = unsafe { raw::mlx_closure_new() };
        Self { raw_closure }
    }

    fn from_function(
        graph_builder: MlxGraphBuilder,
        compile_operation: &'static str,
    ) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The callback has static lifetime and follows the MLX C closure ABI.
        let raw_closure = unsafe { raw::mlx_closure_new_func(Some(graph_builder)) };
        if raw_closure.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: compile_operation,
                description: "MLX returned an empty closure handle".to_owned(),
            });
        }
        Ok(Self { raw_closure })
    }

    const fn raw(&self) -> raw::mlx_closure {
        self.raw_closure
    }

    fn raw_mut(&mut self) -> *mut raw::mlx_closure {
        &mut self.raw_closure
    }

    fn require_populated(&self, compile_operation: &'static str) -> Result<(), MlxRuntimeError> {
        if self.raw_closure.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: compile_operation,
                description: "MLX left the compiled closure handle empty".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxClosure {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live closure handle exactly once.
        unsafe {
            raw::mlx_closure_free(self.raw_closure);
        }
    }
}
