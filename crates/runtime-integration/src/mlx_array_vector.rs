use crate::{MlxArray, MlxRuntimeError, raw};

/// Temporary MLX vector that retains array handles for one aggregate operation.
#[derive(Debug)]
pub(crate) struct MlxArrayVector(raw::mlx_vector_array);

impl MlxArrayVector {
    pub(crate) fn empty(operation: &'static str) -> Result<Self, MlxRuntimeError> {
        // SAFETY: The runtime error handler is installed before model loading,
        // and the official API returns one owned vector handle.
        let raw_vector = unsafe { raw::mlx_vector_array_new() };
        if raw_vector.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }

    pub(crate) fn new(arrays: &[&MlxArray]) -> Result<Self, MlxRuntimeError> {
        let raw_arrays = arrays.iter().map(|array| array.raw()).collect::<Vec<_>>();
        // SAFETY: `raw_arrays` remains valid for this copying constructor, and
        // every handle originates from a live borrowed array owner.
        let raw_vector =
            unsafe { raw::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };
        if raw_vector.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation: "create an MLX array vector",
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }

    pub(crate) const fn raw(&self) -> raw::mlx_vector_array {
        self.0
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_vector_array {
        &mut self.0
    }

    pub(crate) fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_array_size(self.0) }
    }

    pub(crate) fn array_at(
        &self,
        output_index: usize,
        operation: &'static str,
    ) -> Result<MlxArray, MlxRuntimeError> {
        let mut output_array = MlxArray::empty();
        // SAFETY: `self` owns a live vector; `output_array` is uniquely writable;
        // MLX copies the selected array handle into that output owner.
        let status =
            unsafe { raw::mlx_vector_array_get(output_array.raw_mut(), self.0, output_index) };
        crate::mlx_runtime::check_status(status, operation)?;
        output_array.require_populated(operation)?;
        Ok(output_array)
    }
}

impl Drop for MlxArrayVector {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live vector handle exactly once.
        unsafe {
            raw::mlx_vector_array_free(self.0);
        }
    }
}
