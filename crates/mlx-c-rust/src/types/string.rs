//! Owned MLX string handle with the crate's captured-error semantics.

use std::ffi::{CStr, CString};

use crate::MlxCError;
use crate::error::check_status;
use crate::raw;

/// Owned MLX string handle released exactly once through the official C API.
#[derive(Debug)]
pub struct MlxString(raw::mlx_string);

impl MlxString {
    /// Creates an empty string handle for an output parameter.
    pub fn empty() -> Self {
        // SAFETY: The runtime error handler is installed before model loading,
        // and the returned handle enters RAII ownership immediately.
        let raw_string = unsafe { raw::mlx_string_new() };
        Self(raw_string)
    }

    /// Creates an owned string from host UTF-8 text.
    pub fn from_str(value: &str) -> Result<Self, MlxCError> {
        let value_argument = CString::new(value).map_err(|_| MlxCError {
            operation: "create an MLX string",
            description: "string contains an interior null byte".to_owned(),
        })?;
        // SAFETY: The C string remains valid for this copying constructor and
        // the returned handle enters RAII ownership.
        let raw_string = unsafe { raw::mlx_string_new_data(value_argument.as_ptr()) };
        let string = Self(raw_string);
        string.require_populated("create an MLX string")?;
        Ok(string)
    }

    /// Copies the source string handle into this handle.
    pub fn set(&mut self, source: &Self) -> Result<(), MlxCError> {
        // SAFETY: Both handles are live and this call only replaces the owned
        // string value.
        let status = unsafe { raw::mlx_string_set(&mut self.0, source.0) };
        check_status(status, "copy an MLX string")
    }

    /// Copies the owned string text into Rust memory.
    #[must_use]
    pub fn to_string_lossy(&self) -> String {
        // SAFETY: `self` owns a live string handle whose data pointer stays
        // valid for the duration of this call.
        let data_pointer = unsafe { raw::mlx_string_data(self.0) };
        if data_pointer.is_null() {
            return String::new();
        }
        // SAFETY: MLX C documents a null-terminated string owned by the
        // handle; the text is copied before the borrow ends.
        unsafe { CStr::from_ptr(data_pointer) }
            .to_string_lossy()
            .into_owned()
    }

    pub(crate) fn raw_mut(&mut self) -> *mut raw::mlx_string {
        &mut self.0
    }

    pub(crate) fn require_populated(&self, operation: &'static str) -> Result<(), MlxCError> {
        if self.0.ctx.is_null() {
            return Err(MlxCError {
                operation,
                description: "MLX returned an empty string handle".to_owned(),
            });
        }
        Ok(())
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live string exactly once and never
        // accesses the handle afterward.
        unsafe {
            raw::mlx_string_free(self.0);
        }
    }
}
