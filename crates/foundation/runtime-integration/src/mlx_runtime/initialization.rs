//! The one-time construction path of [`MlxRuntime`].
//!
//! # Why initialization is a strict sequence
//!
//! Each step protects the ones after it:
//!
//! 1. The **non-terminating error handler** must be installed before any
//!    fallible MLX call. Stock MLX writes failures to stderr and calls
//!    `std::terminate`; Astronomical captures the description through the
//!    error channel instead so a failed operation becomes a typed
//!    [`MlxRuntimeError`] rather than a dead worker.
//! 2. On macOS, the **Metallib path** is resolved and validated before the
//!    bindings context exists, so the context's first graphics-processor
//!    touch already sees the pinned kernel archive. Linux builds MLX's CPU
//!    backend, which has no kernel archive.
//! 3. On a Linux CUDA build, the **CUDA run-time settings** (the header
//!    folders of MLX's run-time CUDA compiler, and TF32 off) are applied
//!    before MLX can compile a kernel or run a matrix multiply
//!    (`cuda_runtime_settings`).
//! 4. The **memory policy** is applied before any allocation can happen, so
//!    no transient allocation can escape the ceiling the worker promised. On
//!    a CUDA build it first reads the GPU's memory, and MLX holds the limits
//!    capped at it (`cuda_device_memory`).
//! 5. Only then is the **bindings context** created.
//!
//! Reversing any pair would open a window where MLX runs with default policy
//! — which is exactly what the once-per-process statics in `memory_policy`
//! and `metallib` exist to prevent.

#[cfg(target_os = "macos")]
use std::path::Path;

use crate::{MlxMemoryLimits, MlxRuntimeError};
use mlx_bridge::MlxBindingsContext;

#[cfg(target_os = "macos")]
use super::metallib;
use super::{MlxRuntime, cuda_runtime_settings, memory_policy};

impl MlxRuntime {
    /// Installs the non-terminating error handler before any fallible MLX call
    /// and applies the fixed allocator policy once for the worker process.
    ///
    /// # Errors
    ///
    /// Fails when the Metallib path cannot be resolved or validated, when the
    /// memory limits are rejected (including when a *different* configuration
    /// is already active in this process), or when the bindings context
    /// cannot be created.
    pub fn initialize(memory_limits: MlxMemoryLimits) -> Result<Self, MlxRuntimeError> {
        mlx_bridge::install_non_terminating_error_handler();
        #[cfg(target_os = "macos")]
        let metallib_path = metallib::configured_metallib_path()?;
        #[cfg(target_os = "macos")]
        metallib::configure_metallib_path(&metallib_path)?;
        configure_process_settings(memory_limits)?;
        let context = create_bindings_context_for_admitted_limits()?;
        Ok(Self {
            context,
            memory_limits,
            #[cfg(target_os = "macos")]
            metallib_path,
        })
    }

    /// Returns the linked upstream MLX version.
    ///
    /// Surfaced for diagnostics and acceptance contracts that pin behavior to
    /// a specific native release.
    #[must_use]
    pub fn version(&self) -> &str {
        self.context.version()
    }

    /// Returns the memory policy applied during initialization, as
    /// requested. On a CUDA build MLX holds these limits capped at the GPU's
    /// memory; [`MlxRuntime::configured_memory_limit_bytes`](super::memory_policy::MlxRuntime::configured_memory_limit_bytes)
    /// reports what MLX holds (issue #2706).
    ///
    /// This is the policy *this instance* was built with; live updates go
    /// through [`MlxRuntime::update_memory_limits`](super::memory_policy::MlxRuntime::update_memory_limits),
    /// which keeps this value current after a successful swap.
    #[must_use]
    pub const fn memory_limits(&self) -> MlxMemoryLimits {
        self.memory_limits
    }

    /// Returns the absolute AOT Metal library path selected before GPU setup.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn metallib_path(&self) -> &Path {
        &self.metallib_path
    }
}

/// The process-wide settings a runtime needs before its bindings context:
/// MLX's CUDA run-time settings (nothing without the CUDA backend), then this
/// runtime's memory limits, admitted and counted.
fn configure_process_settings(memory_limits: MlxMemoryLimits) -> Result<(), MlxRuntimeError> {
    cuda_runtime_settings::configure_cuda_runtime_settings()?;
    memory_policy::configure_runtime_memory_limits(memory_limits)
}

/// Creates the bindings context after the memory limits were admitted and
/// counted. If creation fails no runtime will exist to release that count on
/// drop, so it is released here before the error returns.
fn create_bindings_context_for_admitted_limits() -> Result<MlxBindingsContext, MlxRuntimeError> {
    platform_bindings_context().map_err(release_admitted_limits_on_failure)
}

/// A bindings context on the device this build of MLX computes on: the GPU
/// stream on macOS, exactly as before, and elsewhere the GPU when MLX has
/// one or the CPU when it has none. Every caller that builds its own MLX
/// graph context goes through this, so no graph asks a GPU-less MLX for a
/// GPU stream.
///
/// # Errors
///
/// Returns the MLX description when the stream cannot be acquired.
#[cfg(target_os = "macos")]
pub fn platform_bindings_context() -> Result<MlxBindingsContext, mlx_bridge::MlxCError> {
    MlxBindingsContext::new()
}

/// Elsewhere MLX may have been built with a GPU backend (CUDA) or with none
/// (the CPU backend Linux builds today, issue #2424). The GPU stream is used
/// whenever MLX has a GPU; otherwise the runtime runs on MLX's CPU stream,
/// because asking a GPU-less MLX for its GPU stream fails.
///
/// # Errors
///
/// Returns the MLX description when the stream cannot be acquired.
#[cfg(not(target_os = "macos"))]
pub fn platform_bindings_context() -> Result<MlxBindingsContext, mlx_bridge::MlxCError> {
    if mlx_bridge::metal_is_available()? || mlx_bridge::cuda_is_available()? {
        MlxBindingsContext::new()
    } else {
        MlxBindingsContext::on_cpu()
    }
}

fn release_admitted_limits_on_failure(context_error: mlx_bridge::MlxCError) -> MlxRuntimeError {
    memory_policy::release_runtime_memory_limits();
    MlxRuntimeError::from(context_error)
}

impl Drop for MlxRuntime {
    /// Ends this runtime's hold on the process memory policy, so once no
    /// runtime is alive the next one may install different limits (#1906).
    fn drop(&mut self) {
        memory_policy::release_runtime_memory_limits();
    }
}
