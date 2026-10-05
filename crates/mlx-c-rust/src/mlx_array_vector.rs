use crate::raw;
use crate::{MlxArray, MlxCError, error::check_status};

/// Temporary MLX vector that retains array handles for one aggregate operation.
#[derive(Debug)]
pub struct MlxArrayVector(raw::mlx_vector_array);

impl MlxArrayVector {
    pub fn empty(operation: &'static str) -> Result<Self, MlxCError> {
        // SAFETY: The runtime error handler is installed before model loading,
        // and the official API returns one owned vector handle.
        let raw_vector = unsafe { raw::mlx_vector_array_new() };
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }

    pub fn new(arrays: &[&MlxArray]) -> Result<Self, MlxCError> {
        let raw_arrays = arrays.iter().map(|array| array.raw()).collect::<Vec<_>>();
        // SAFETY: `raw_arrays` remains valid for this copying constructor, and
        // every handle originates from a live borrowed array owner.
        let raw_vector =
            unsafe { raw::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation: "create an MLX array vector",
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }

    pub const fn raw(&self) -> raw::mlx_vector_array {
        self.0
    }

    pub fn raw_mut(&mut self) -> *mut raw::mlx_vector_array {
        &mut self.0
    }

    pub(crate) fn from_live_raw(
        raw_vector: raw::mlx_vector_array,
        operation: &'static str,
    ) -> Result<Self, MlxCError> {
        if raw_vector.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty vector handle".to_owned(),
            });
        }
        Ok(Self(raw_vector))
    }

    pub fn len(&self) -> usize {
        // SAFETY: `self` owns a live MLX vector handle.
        unsafe { raw::mlx_vector_array_size(self.0) }
    }

    pub fn array_at(
        &self,
        output_index: usize,
        operation: &'static str,
    ) -> Result<MlxArray, MlxCError> {
        let mut output_array = MlxArray::empty();
        // SAFETY: `self` owns a live vector; `output_array` is uniquely writable;
        // MLX copies the selected array handle into that output owner.
        let status =
            unsafe { raw::mlx_vector_array_get(output_array.raw_mut(), self.0, output_index) };
        check_status(status, operation)?;
        output_array.require_populated(operation)?;
        Ok(output_array)
    }

    /// Copies the source vector handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the owned
        // vector value.
        let status = unsafe { raw::mlx_vector_array_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX array vector")
    }

    /// Creates an owned vector holding one copied array handle.
    pub fn from_array(array: &MlxArray) -> Result<Self, MlxCError> {
        // SAFETY: The array handle is live and copied into the new vector.
        let raw_vector = unsafe { raw::mlx_vector_array_new_value(array.raw()) };
        Self::from_live_raw(raw_vector, "create an MLX array vector")
    }

    /// Replaces the vector contents with copies of the provided arrays.
    pub fn set_arrays(&mut self, arrays: &[MlxArray]) -> Result<(), MlxCError> {
        let raw_arrays: Vec<raw::mlx_array> = arrays.iter().map(MlxArray::raw).collect();
        // SAFETY: The handle storage remains valid for this copying call.
        let status = unsafe {
            raw::mlx_vector_array_set_data(&mut self.0, raw_arrays.as_ptr(), raw_arrays.len())
        };
        check_status(status, "replace MLX array vector contents")
    }

    /// Replaces the vector contents with one copied array handle.
    pub fn set_one(&mut self, array: &MlxArray) -> Result<(), MlxCError> {
        // SAFETY: The array handle is live and copied into the vector.
        let status = unsafe { raw::mlx_vector_array_set_value(&mut self.0, array.raw()) };
        check_status(status, "replace MLX array vector contents")
    }

    /// Appends one copied array handle.
    pub fn append_array(&mut self, array: &MlxArray) -> Result<(), MlxCError> {
        // SAFETY: The array handle is live and copied into the vector.
        let status = unsafe { raw::mlx_vector_array_append_value(self.0, array.raw()) };
        check_status(status, "append to an MLX array vector")
    }

    /// Appends copies of the provided array handles.
    pub fn append_arrays(&mut self, arrays: &[MlxArray]) -> Result<(), MlxCError> {
        let raw_arrays: Vec<raw::mlx_array> = arrays.iter().map(MlxArray::raw).collect();
        // SAFETY: The handle storage remains valid for this copying call.
        let status = unsafe {
            raw::mlx_vector_array_append_data(self.0, raw_arrays.as_ptr(), raw_arrays.len())
        };
        check_status(status, "append to an MLX array vector")
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
