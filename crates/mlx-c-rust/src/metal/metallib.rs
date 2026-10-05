//! Safe wrapper over the MLX AOT metallib path override.

use std::ffi::c_char;

use crate::{MlxCError, error::check_status, raw};

/// Overrides the metallib MLX loads, from a NUL-terminated path.
///
/// The policy owner resolves and verifies WHICH metallib to load; this wrapper
/// only performs the official override call.
///
/// # Safety for callers
/// The path must remain valid (NUL-terminated, not dangling) for the duration
/// of this synchronous call.
pub fn set_metallib_path(nul_terminated_path: *const c_char) -> Result<(), MlxCError> {
    // SAFETY: The official C API copies the non-null NUL-terminated path and
    // does not retain the borrowed pointer after returning.
    let status = unsafe { raw::mlx_metal_set_metallib_path(nul_terminated_path) };
    check_status(status, "set the MLX AOT metallib path")
}
