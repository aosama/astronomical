use std::path::Path;

use crate::{MlxMemoryLimits, MlxRuntimeError};
use astronomical_mlx_c_rust::{MlxBindingsContext, install_non_terminating_error_handler};

use super::{MlxRuntime, memory_policy, metallib};

impl MlxRuntime {
    /// Installs the non-terminating error handler before any fallible MLX call
    /// and applies the fixed allocator policy once for the worker process.
    pub fn initialize(memory_limits: MlxMemoryLimits) -> Result<Self, MlxRuntimeError> {
        install_non_terminating_error_handler();
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
    #[must_use]
    pub fn version(&self) -> &str {
        self.context.version()
    }

    /// Returns the memory policy applied during initialization.
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
