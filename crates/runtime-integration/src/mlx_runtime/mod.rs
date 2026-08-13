mod error_handling;
mod initialization;
mod memory_policy;
mod metallib;
mod safetensors;
mod version;

use std::path::PathBuf;

use crate::{MlxMemoryLimits, mlx_stream::MlxStream};

pub use error_handling::classify_mlx_error;
pub(crate) use error_handling::{
    check_status, clear_captured_mlx_error, install_non_terminating_error_handler,
    take_captured_mlx_error,
};
pub use metallib::{compiled_metallib_path, validate_metallib_path};
pub(crate) use metallib::{configure_metallib_path, configured_metallib_path};

/// Process-global official MLX C runtime configured for one isolated worker.
#[derive(Debug)]
pub struct MlxRuntime {
    gpu_stream: MlxStream,
    memory_limits: MlxMemoryLimits,
    metallib_path: PathBuf,
    version: String,
}
