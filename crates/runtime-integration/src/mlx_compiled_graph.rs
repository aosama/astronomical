use std::os::raw::c_int;

use crate::{
    MlxArray, MlxRuntimeError, mlx_array_vector::MlxArrayVector, mlx_runtime::check_status, raw,
};

pub(crate) type MlxGraphBuilder =
    unsafe extern "C" fn(*mut raw::mlx_vector_array, raw::mlx_vector_array) -> c_int;

/// Owns one compiled MLX graph with a single array output.
#[derive(Debug)]
pub(crate) struct MlxCompiledGraph {
    compiled_closure: MlxClosure,
}

impl MlxCompiledGraph {
    pub(crate) fn new(
        graph_builder: MlxGraphBuilder,
        compile_operation: &'static str,
    ) -> Result<Self, MlxRuntimeError> {
        let source_closure = MlxClosure::from_function(graph_builder, compile_operation)?;
        let mut compiled_closure = MlxClosure::empty();
        // SAFETY: Both closure handles are live and uniquely owned. MLX copies
        // the compiled function into `compiled_closure`.
        let compile_status =
            unsafe { raw::mlx_compile(compiled_closure.raw_mut(), source_closure.raw(), true) };
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
}

pub(crate) fn array_from_vector(
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

pub(crate) fn graph_output_array(
    build_graph: impl FnOnce(*mut raw::mlx_array) -> c_int,
) -> Result<MlxArray, c_int> {
    let mut output_array = MlxArray::empty();
    let build_status = build_graph(output_array.raw_mut());
    if build_status != 0 || output_array.is_empty() {
        return Err(if build_status == 0 { 1 } else { build_status });
    }
    Ok(output_array)
}

pub(crate) unsafe fn set_graph_output(
    output_vector: *mut raw::mlx_vector_array,
    graph_output: &MlxArray,
) -> c_int {
    // SAFETY: The caller guarantees that the destination vector is unique and
    // live. MLX copies the lazy output handle before local owners are released.
    unsafe { raw::mlx_vector_array_set_value(output_vector, graph_output.raw()) }
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
