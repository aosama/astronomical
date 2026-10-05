mod error_handling;
mod initialization;
mod memory_policy;
mod metallib;
mod safetensors;

use std::ops::Deref;
use std::path::PathBuf;

use crate::MlxMemoryLimits;
use astronomical_mlx_c_rust::MlxBindingsContext;

pub(crate) use error_handling::check_status;
pub use error_handling::classify_mlx_error;
pub use metallib::{compiled_metallib_path, validate_metallib_path};
pub(crate) use metallib::{configure_metallib_path, configured_metallib_path};

/// Process-global official MLX C runtime configured for one isolated worker.
///
/// The runtime is Astronomical policy over the bindings context: it owns the
/// memory-limit enforcement and the metallib path selection, and it composes
/// the bindings context (`MlxBindingsContext`) that carries the GPU stream and
/// the linked upstream version. Operation wrappers live on the context; this
/// dereferences to it so policy and operations read as one runtime surface.
#[derive(Debug)]
pub struct MlxRuntime {
    context: MlxBindingsContext,
    memory_limits: MlxMemoryLimits,
    metallib_path: PathBuf,
}

impl Deref for MlxRuntime {
    type Target = MlxBindingsContext;

    /// Deliberate ownership deref, not a compatibility alias: the policy owner
    /// IS a configured bindings context plus Astronomical policy, so operation
    /// calls read identically on both sides of the ownership boundary.
    fn deref(&self) -> &Self::Target {
        &self.context
    }
}
