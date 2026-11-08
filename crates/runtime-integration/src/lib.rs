//! Narrow unsafe ownership boundary around the official MLX C API.
//!
//! # The one job of this crate
//!
//! Every direct touch of the MLX C API in Astronomical happens here. Code
//! above this crate — the model engines, the daemon, the quantization tools —
//! works exclusively with the safe, typed surface re-exported from this
//! module. Concentrating the unsafe surface here is what lets the rest of the
//! workspace compile with `unsafe_code = "deny"` and lets every `unsafe` block
//! in the product be audited by reading one crate.
//!
//! # The two halves
//!
//! The crate owns two fundamentally different jobs, and the file layout keeps
//! them apart:
//!
//! 1. **Native build orchestration** — `build.rs` plus the modules under
//!    `build-orchestration/` compile the pinned, patched MLX native image
//!    through CMake, publish it into a machine-wide identity-keyed store, and
//!    teach Cargo how to link it. These files run at build time, before any
//!    Rust in this crate exists. See `build.rs` for that half's module map.
//! 2. **The runtime boundary** — the modules under `src/` wrap the compiled
//!    image at process run time: process-global policy (memory ceilings, the
//!    Metal library selection), typed error classification, and the custom
//!    safetensors input/output engines.
//!
//! # Module map of the runtime half
//!
//! - [`mlx_metallib_path`] — resolves which Metal library (metallib) file a
//!   worker should open. Deliberately NOT behind the `mlx` feature so its
//!   resolution rules stay testable without the native image.
//! - [`mlx_runtime_types`] — the typed vocabulary of the boundary (memory
//!   limits, memory snapshots, errors, reclaim thresholds). Also ungated:
//!   lightweight metadata types must not drag the graphics-processor build
//!   into consumers that never link MLX.
//! - [`mlx_runtime`] — [`MlxRuntime`], the process-global policy owner: one
//!   installed error handler, one Metallib path, one pair of memory limits
//!   per worker process, plus live-limit updates, allocator-cache reclamation,
//!   and GPU-stream synchronization.
//! - [`safetensors`] — the custom C-ABI input/output engines handed to MLX:
//!   the retained-descriptor reader for whole-file weight loading, the
//!   bounded multi-range reader that expert paging is built on, the
//!   descriptor-backed and in-memory writers used by the persistent prompt
//!   cache and the quantization tools, and the positional-read metrics that
//!   make the input/output observable.
//!
//! # Process-global invariants (why the statics exist)
//!
//! One worker process serves exactly one MLX runtime configuration. MLX-C
//! itself is process-global: its error handler, Metallib path, and memory
//! limits are installed once for the process, not per object. The statics in
//! `mlx_runtime` (`RUNTIME_MEMORY_LIMITS`, `RUNTIME_METALLIB_PATH`) therefore
//! enforce once-per-process semantics and reject a second, different
//! configuration instead of silently racing the native state. This mirrors the
//! product model: the daemon spawns one worker per model load, and a model
//! swap replaces the worker process rather than reconfiguring MLX in place.
//!
//! # Relationship to the bridge crates
//!
//! The stock MLX-C surface is bound by the published `mlx-c-rust` crate. The
//! handful of Astronomical-specific native symbols added by the patch set
//! under `third-party/patches` are bound by `astronomical-mlx-c-fork`. This
//! crate composes both: operations come through `MlxRuntime`'s dereference to
//! `MlxBindingsContext`, Astronomical policy lives in the modules here.
//!
//! # Feature gating
//!
//! Everything that needs the compiled native image is behind the `mlx`
//! feature. The two ungated modules exist so packaging scripts and tests can
//! reason about Metallib resolution and boundary types on any machine — for
//! example a container lane that only checks formatting or documentation.

mod mlx_metallib_path;
mod mlx_runtime_types;
#[cfg(test)]
mod mlx_runtime_types_tests;

#[cfg(feature = "mlx")]
mod mlx_runtime;
#[cfg(feature = "mlx")]
mod safetensors;

pub use mlx_metallib_path::resolve_mlx_metallib_path;
pub use mlx_runtime_types::{
    ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES, MlxMemoryLimits, MlxMemorySnapshot, MlxRuntimeError,
    allocator_cache_exceeds_reclaim_threshold,
};

#[cfg(feature = "mlx")]
pub use mlx_runtime::{
    MlxRuntime, classify_mlx_error, compiled_metallib_path,
    maximum_recommended_gpu_working_set_size_bytes, validate_metallib_path,
};
#[cfg(feature = "mlx")]
pub use safetensors::file_writer::{MlxSafetensorsWriteOutcome, MlxSafetensorsWriterError};
#[cfg(feature = "mlx")]
pub use safetensors::lazy_map::{BoundedReadInterval, MlxSafetensors, SafetensorsLoadResult};
#[cfg(feature = "mlx")]
pub use safetensors::read_metrics::{PositionalFileReadMetrics, PositionalFileReadMetricsSnapshot};
