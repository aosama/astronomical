//! [`MlxRuntime`]: the process-global official MLX C runtime configured for
//! one isolated worker.
//!
//! # Module map
//!
//! - `initialization` — the one-time construction path: error handler,
//!   Metallib path, memory policy, bindings context.
//! - `memory_policy` — the live memory-limit surface: updates with rollback,
//!   snapshots, peak reset, allocator-cache reclamation, stream
//!   synchronization.
//! - `error_classification` — turns captured MLX-C error descriptions into
//!   Astronomical's typed errors, including the recoverable
//!   active-memory-ceiling classification.
//! - `metallib` — resolves, validates, and installs the Ahead-Of-Time Metal
//!   library path exactly once per process.
//! - `device_info` — reads Metal's advertised recommended working-set size
//!   through MLX-C (diagnostic surface only).
//! - `safetensors_io` — the runtime-level entry points that hand the custom
//!   input/output engines under `crate::safetensors` to callers.

mod error_classification;
mod initialization;
mod memory_policy;
#[cfg(test)]
mod memory_policy_tests;
mod metallib;
mod safetensors_io;

use std::ops::Deref;
use std::path::PathBuf;

use crate::MlxMemoryLimits;
use mlx_c_rust::MlxBindingsContext;

pub use error_classification::classify_mlx_error;
pub use metallib::{compiled_metallib_path, validate_metallib_path};
pub(crate) use metallib::{configure_metallib_path, configured_metallib_path};

// Re-exported for the crate root: the device probe is part of the runtime's
// public diagnostic surface even though it never needs a full MlxRuntime.
pub use device_info::maximum_recommended_gpu_working_set_size_bytes;
mod device_info;

// The status-check helper and the poison-tolerant lock helper are shared by
// every sibling module here, so they travel through the module tree rather
// than being repeated.
pub(crate) use error_classification::check_status;

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
