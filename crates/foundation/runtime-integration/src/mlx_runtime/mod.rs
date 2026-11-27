//! [`MlxRuntime`]: the process-global official MLX C runtime configured for
//! one isolated worker.
//!
//! # Module map
//!
//! - `initialization` — the one-time construction path: error handler,
//!   Metallib path, memory policy, bindings context.
//! - `memory_policy` — the live memory-limit surface: updates with rollback,
//!   snapshots, peak reset, MLX buffer-cache reclamation, stream
//!   synchronization.
//! - `error_classification` — turns captured MLX-C error descriptions into
//!   Astronomical's typed errors, including the recoverable
//!   active-memory-ceiling classification.
//! - `metallib` — resolves, validates, and installs the Ahead-Of-Time Metal
//!   library path exactly once per process (macOS only).
//! - `device_info` — reads Metal's advertised recommended working-set size
//!   through MLX-C (diagnostic surface only; macOS only).
//! - `safetensors_io` — the runtime-level entry points that hand the custom
//!   input/output engines under `crate::safetensors` to callers.

mod cuda_runtime_settings;
mod error_classification;
mod initialization;
pub use initialization::platform_bindings_context;
mod memory_policy;
#[cfg(test)]
mod memory_policy_tests;
// The Metallib exists only for the Metal backend; a Linux build runs MLX's
// CPU backend and has no kernel archive to resolve or verify (issue #2424).
#[cfg(target_os = "macos")]
mod metallib;
mod quantized_projection;
pub use quantized_projection::mlx_computes_on_cpu;
mod safetensors_io;

use std::ops::Deref;
#[cfg(target_os = "macos")]
use std::path::PathBuf;

use crate::MlxMemoryLimits;
use mlx_bridge::MlxBindingsContext;

pub use error_classification::classify_mlx_error;
#[cfg(target_os = "macos")]
pub use metallib::{compiled_metallib_path, validate_metallib_path};
#[cfg(target_os = "macos")]
pub(crate) use metallib::{configure_metallib_path, configured_metallib_path};

// Re-exported for the crate root: the device probe is part of the runtime's
// public diagnostic surface even though it never needs a full MlxRuntime.
pub use array_storage::evaluated_array_storage_mut;
// Metal's recommended working set is the only device figure the runtime
// reads, and a Linux build has no GPU device to ask, so the probe exists on
// macOS only; callers there get a compile error rather than a run-time MLX
// "no GPU" failure.
#[cfg(target_os = "macos")]
pub use device_info::maximum_recommended_gpu_working_set_size_bytes;
mod array_storage;
#[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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
