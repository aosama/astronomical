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
//! 2. The **Metallib path** is resolved and validated before the bindings
//!    context exists, so the context's first graphics-processor touch already
//!    sees the pinned kernel archive.
//! 3. The **memory policy** is applied before any allocation can happen, so
//!    no transient allocation can escape the ceiling the worker promised.
//! 4. Only then is the **bindings context** created.
//!
//! Reversing any pair would open a window where MLX runs with default policy
//! — which is exactly what the once-per-process statics in `memory_policy`
//! and `metallib` exist to prevent.

use std::path::Path;

use crate::{MlxMemoryLimits, MlxRuntimeError};
use mlx_c_rust::MlxBindingsContext;

use super::{MlxRuntime, memory_policy, metallib};

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
        mlx_c_rust::install_non_terminating_error_handler();
        let metallib_path = metallib::configured_metallib_path()?;
        metallib::configure_metallib_path(&metallib_path)?;
        memory_policy::configure_runtime_memory_limits(memory_limits)?;
        let context = MlxBindingsContext::new().map_err(MlxRuntimeError::from)?;
        Ok(Self {
            context,
            memory_limits,
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

    /// Returns the memory policy applied during initialization.
    ///
    /// This is the policy *this instance* was built with; live updates go
    /// through [`MlxRuntime::update_memory_limits`](super::memory_policy::MlxRuntime::update_memory_limits),
    /// which keeps this value current after a successful swap.
    #[must_use]
    pub const fn memory_limits(&self) -> MlxMemoryLimits {
        self.memory_limits
    }

    /// Returns the absolute AOT Metal library path selected before GPU setup.
    #[must_use]
    pub fn metallib_path(&self) -> &Path {
        &self.metallib_path
    }
}
