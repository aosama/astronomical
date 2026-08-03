use std::path::Path;

use crate::{MlxMemoryLimits, MlxRuntimeError, mlx_stream::MlxStream};

use super::{MlxRuntime, error_handling, memory_policy, metallib, version};

impl MlxRuntime {
    /// Installs the non-terminating error handler before any fallible MLX call
    /// and applies the fixed allocator policy once for the worker process.
    pub fn initialize(memory_limits: MlxMemoryLimits) -> Result<Self, MlxRuntimeError> {
        error_handling::install_non_terminating_error_handler();
        let metallib_path = metallib::configured_metallib_path()?;
        metallib::configure_metallib_path(&metallib_path)?;
        memory_policy::configure_runtime_memory_limits(memory_limits)?;
        let version = version::read_mlx_version()?;
        let gpu_stream = MlxStream::default_gpu()?;
        Ok(Self {
            gpu_stream,
            memory_limits,
            metallib_path,
            version,
        })
    }

    /// Returns the linked upstream MLX version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
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
