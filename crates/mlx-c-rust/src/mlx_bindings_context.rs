//! The per-worker MLX bindings context: the GPU stream and the linked version.
//!
//! This is the split the runtime-owner issue anticipates: the context holds
//! exactly the bindings-owned state (which stream work is submitted to, which
//! upstream version is linked), while Astronomical policy — memory limits and
//! the metallib path selection — stays with the policy owner in
//! `astronomical-runtime-integration`, which composes this context.

use std::ffi::CStr;

use crate::{MlxCError, MlxStream, error::check_status, raw};

/// Process-global official MLX C bindings context configured for one worker.
#[derive(Debug)]
pub struct MlxBindingsContext {
    gpu_stream: MlxStream,
    version: String,
}

impl MlxBindingsContext {
    /// Creates the context after the non-terminating error handler is
    /// installed: reads the linked version and acquires the default GPU stream.
    pub fn new() -> Result<Self, MlxCError> {
        let version = read_mlx_version()?;
        let gpu_stream = MlxStream::default_gpu()?;
        Ok(Self {
            gpu_stream,
            version,
        })
    }

    /// The GPU stream every worker submission targets.
    pub const fn gpu_stream(&self) -> &MlxStream {
        &self.gpu_stream
    }

    /// The linked upstream MLX version.
    pub fn version(&self) -> &str {
        &self.version
    }
}

fn read_mlx_version() -> Result<String, MlxCError> {
    // SAFETY: The error handler is installed before this constructor and the
    // returned owned handle is released on every subsequent path.
    let raw_version = unsafe { raw::mlx_string_new() };
    let mut owned_version = OwnedMlxString(raw_version);
    // SAFETY: `owned_version` contains a live MLX string handle and this call
    // only replaces its owned string value.
    let status = unsafe { raw::mlx_version(&mut owned_version.0) };
    check_status(status, "read the linked MLX version")?;
    // SAFETY: The pointer remains valid while `owned_version` is live.
    let version_pointer = unsafe { raw::mlx_string_data(owned_version.0) };
    if version_pointer.is_null() {
        return Err(MlxCError {
            operation: "read the linked MLX version",
            description: "MLX returned a null version string".to_owned(),
        });
    }
    // SAFETY: MLX C documents a null-terminated string owned by the handle.
    Ok(unsafe { CStr::from_ptr(version_pointer) }
        .to_string_lossy()
        .into_owned())
}

struct OwnedMlxString(raw::mlx_string);

impl Drop for OwnedMlxString {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live handle exactly once and does not
        // use it afterward. MLX C accepts the value form for destruction.
        unsafe {
            raw::mlx_string_free(self.0);
        }
    }
}
