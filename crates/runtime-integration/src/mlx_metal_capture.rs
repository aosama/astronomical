//! On-demand MLX-C Metal GPU capture control.
//!
//! C declarations: `mlx-c/mlx/c/metal.h::{mlx_metal_start_capture,
//! mlx_metal_stop_capture}`. The bindgen allowlist does not include these two
//! symbols, so they are declared here directly against the already-linked
//! `mlxc`/`mlx` native libraries.
//!
//! Capture is opt-in and inert by default. Metal only records when the target
//! process was started with `MTL_CAPTURE_ENABLED=1`; without that layer,
//! `mlx_metal_start_capture` fails and the caller logs and continues, so
//! enabling the env var can never change a serving request's result.

use std::ffi::{CString, c_char, c_int};

use crate::{MlxRuntime, MlxRuntimeError, mlx_runtime::check_status};

unsafe extern "C" {
    fn mlx_metal_start_capture(path: *const c_char) -> c_int;
    fn mlx_metal_stop_capture() -> c_int;
}

impl MlxRuntime {
    /// Starts a Metal GPU capture writing an Xcode `.gputrace` bundle at `capture_path`.
    pub fn start_metal_capture(&self, capture_path: &str) -> Result<(), MlxRuntimeError> {
        const OPERATION: &str = "start an MLX Metal capture";
        let capture_path =
            CString::new(capture_path).map_err(|source| MlxRuntimeError::RuntimeOperation {
                operation: OPERATION,
                description: format!("capture path contains an interior NUL byte: {source}"),
            })?;
        // SAFETY: The path is a live NUL-terminated string for the duration of
        // this call; MLX copies it before returning.
        let status = unsafe { mlx_metal_start_capture(capture_path.as_ptr()) };
        check_status(status, OPERATION)
    }

    /// Stops the active Metal GPU capture, finalizing the `.gputrace` bundle.
    pub fn stop_metal_capture(&self) -> Result<(), MlxRuntimeError> {
        const OPERATION: &str = "stop an MLX Metal capture";
        // SAFETY: The runtime owns the live GPU stream for the worker lifetime;
        // stopping a capture is safe and idempotent with respect to that stream.
        let status = unsafe { mlx_metal_stop_capture() };
        check_status(status, OPERATION)
    }
}
