//! The typed vocabulary of the MLX boundary: memory limits, memory snapshots,
//! error taxonomy, and the allocator-cache reclaim policy constants.
//!
//! # Why this module is not behind the `mlx` feature
//!
//! These types describe the contract of the boundary without touching the
//! native image. Keeping them ungated means consumers that only need to talk
//! *about* MLX policy (configuration plumbing, tests, packaging checks) never
//! link the graphics-processor runtime or trigger its native build. The rule
//! mirrors the separation in `native-identity`: identity and policy vocabulary
//! stay lightweight; execution stays behind the feature gate.
//!
//! # Who consumes these types
//!
//! - `MlxMemoryLimits` is constructed by the inference worker and every
//!   model-serving engine at runtime initialization, and swapped at runtime
//!   through [`MlxRuntime::update_memory_limits`] when the user moves the
//!   live ceiling control.
//! - `MlxMemorySnapshot` feeds the Observatory memory attribution and the
//!   throughput-history journeys.
//! - `MlxRuntimeError::ActiveMemoryLimitExceeded` is the typed, recoverable
//!   signal that drives adaptive prefill chunk shrinking in the qwen3.5
//!   engines; `is_recoverable_graphics_processor_out_of_memory` drives the
//!   graphics-processor out-of-memory recovery path.

use std::path::PathBuf;

use thiserror::Error;

/// Memory controls applied inside one MLX worker process.
///
/// These values change no kernel setting; both are process-local MLX
/// guidance. The active-memory value is reused as the process residency-set
/// limit because worker startup already resolved it against the machine
/// ceiling. The allocator-cache value independently controls reclaimable
/// allocation reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemoryLimits {
    pub(crate) active_memory_limit_bytes: usize,
    pub(crate) allocator_cache_memory_limit_bytes: usize,
}

impl MlxMemoryLimits {
    /// Validates explicit active memory and allocator-cache byte limits.
    ///
    /// # Errors
    ///
    /// - The active limit must be positive: a zero limit would make every
    ///   allocation request fail the ceiling check.
    /// - The allocator-cache limit cannot exceed the active limit — the cache
    ///   holds buffers reclaimed from the active set, so a cache allowed to
    ///   retain more than the process may ever hold is incoherent.
    ///
    /// MLX treats a zero cache limit as "do not retain freed allocations",
    /// which is why zero is accepted here.
    pub fn new(
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    ) -> Result<Self, MlxRuntimeError> {
        if active_memory_limit_bytes == 0 {
            return Err(MlxRuntimeError::InvalidMemoryLimits {
                description: "active memory limit must be positive",
            });
        }
        if allocator_cache_memory_limit_bytes > active_memory_limit_bytes {
            return Err(MlxRuntimeError::InvalidMemoryLimits {
                description: "allocator cache memory limit cannot exceed the active memory limit",
            });
        }
        Ok(Self {
            active_memory_limit_bytes,
            allocator_cache_memory_limit_bytes,
        })
    }

    /// Returns MLX's graph-evaluation memory guideline in bytes.
    ///
    /// Upstream MLX documents this value as guidance rather than an allocation
    /// ceiling, so callers must still perform their own admission.
    #[must_use]
    pub const fn active_memory_limit_bytes(self) -> usize {
        self.active_memory_limit_bytes
    }

    /// Returns the allocator cache-reclamation threshold in bytes.
    #[must_use]
    pub const fn allocator_cache_memory_limit_bytes(self) -> usize {
        self.allocator_cache_memory_limit_bytes
    }

    /// Returns the active-memory ceiling plus the approved one-percent transient allowance.
    ///
    /// The patched native allocator (see
    /// `third-party/patches/mlx-0.32.4-astronomical-active-memory-ceiling.patch`)
    /// tolerates this small overshoot before rejecting an allocation, because
    /// graph evaluation routinely needs one transient scratch buffer slightly
    /// above the steady-state ceiling; the error report it emits then carries
    /// this same allowed value, keeping the engine's recovery arithmetic and
    /// the allocator's rejection arithmetic anchored to one number.
    #[must_use]
    pub const fn allowed_active_memory_bytes(self) -> usize {
        let transient_allowance_bytes = self.active_memory_limit_bytes / 100;
        match self
            .active_memory_limit_bytes
            .checked_add(transient_allowance_bytes)
        {
            Some(allowed_active_memory_bytes) => allowed_active_memory_bytes,
            None => usize::MAX,
        }
    }
}

/// One point-in-time process MLX memory observation.
///
/// Read without synchronizing the graphics-processor stream, so a snapshot is
/// cheap enough to sample per telemetry tick; values may lag in-flight work
/// by whatever the stream still has queued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemorySnapshot {
    pub(crate) active_memory_bytes: usize,
    pub(crate) allocator_cache_memory_bytes: usize,
    pub(crate) peak_memory_bytes: usize,
}

/// Reclaimable allocator-cache bytes that justify an IOGPU-visible clear.
///
/// Performance conclusion memorialized from direct-MLX streaming
/// measurements: clearing after every prefill chunk can stall WindowServer
/// (the clear is visible to the graphics subsystem), while leaving the cache
/// unbounded reproduced a 20 GB throughput cliff. This threshold is a
/// portable byte floor, not a machine-specific RAM fraction, so the policy
/// behaves identically on every Mac.
pub const ALLOCATOR_CACHE_RECLAIM_THRESHOLD_BYTES: usize = 256_000_000;

/// Returns whether a synchronized allocator cache is large enough to reclaim.
///
/// Boundary semantics: exactly at the threshold counts as reclaimable. The
/// beside-source tests pin that boundary so a future refactor cannot quietly
/// strand the cache at the policy edge.
#[must_use]
pub const fn allocator_cache_exceeds_reclaim_threshold(
    allocator_cache_memory_bytes: usize,
    reclaim_threshold_bytes: usize,
) -> bool {
    allocator_cache_memory_bytes >= reclaim_threshold_bytes
}

impl MlxMemorySnapshot {
    /// Returns bytes held by live MLX arrays and graphs.
    #[must_use]
    pub const fn active_memory_bytes(self) -> usize {
        self.active_memory_bytes
    }

    /// Returns bytes retained in the MLX allocator cache.
    #[must_use]
    pub const fn allocator_cache_memory_bytes(self) -> usize {
        self.allocator_cache_memory_bytes
    }

    /// Returns peak active allocator bytes since the last reset.
    #[must_use]
    pub const fn peak_memory_bytes(self) -> usize {
        self.peak_memory_bytes
    }
}

/// Typed failures returned by the official MLX C runtime boundary.
///
/// The variants encode what Astronomical can *do* about a failure, not just
/// what failed: [`MlxRuntimeError::ActiveMemoryLimitExceeded`] and the
/// graphics-processor out-of-memory detection on
/// [`MlxRuntimeError::RuntimeOperation`] mark recoverable conditions the
/// engines respond to by shrinking work and retrying; everything else is a
/// hard failure for the request or the worker.
#[derive(Debug, Error)]
pub enum MlxRuntimeError {
    #[error("invalid MLX memory limits: {description}")]
    InvalidMemoryLimits { description: &'static str },
    #[error(
        "MLX runtime already uses active limit {active_memory_limit_bytes} bytes and allocator cache limit {allocator_cache_memory_limit_bytes} bytes"
    )]
    RuntimeAlreadyConfigured {
        active_memory_limit_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    },
    #[error(
        "MLX active memory ceiling rejected allocation: active={active_memory_bytes} attempted={attempted_allocation_bytes} allowed={allowed_active_memory_bytes}"
    )]
    ActiveMemoryLimitExceeded {
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
    },
    #[error(
        "MLX safetensors serialization requires {attempted_serialized_byte_count} bytes, above the permitted {maximum_serialized_byte_count} bytes"
    )]
    SafetensorsSerializationLimitExceeded {
        attempted_serialized_byte_count: usize,
        maximum_serialized_byte_count: usize,
    },
    #[error("invalid MLX AOT metallib path: {description}")]
    InvalidMetallibPath { description: String },
    #[error("MLX runtime already uses AOT metallib {configured_path:?}")]
    MetallibAlreadyConfigured { configured_path: PathBuf },
    #[error("failed to {operation}: {description}")]
    RuntimeOperation {
        operation: &'static str,
        description: String,
    },
}

#[cfg(feature = "mlx")]
impl From<mlx_c_rust::MlxCError> for MlxRuntimeError {
    fn from(captured_error: mlx_c_rust::MlxCError) -> Self {
        // Classification happens at this conversion so every captured MLX-C
        // failure crossing into the runtime boundary — regardless of which
        // layer observed it — is tested against the allocator patch's active
        // memory ceiling marker before falling back to a plain operation
        // failure.
        crate::mlx_runtime::classify_mlx_error(captured_error.operation, captured_error.description)
    }
}

impl MlxRuntimeError {
    /// Returns whether Metal completed a command buffer after exhausting GPU memory.
    ///
    /// Metal reports this condition as a plain error string rather than a
    /// dedicated status code, so the detection matches the two fragments
    /// Metal always includes: the human-readable "Insufficient Memory"
    /// sentence and the `kIOGPU...` kernel error code. Both are required so
    /// an unrelated "Insufficient Memory" failure from another subsystem can
    /// never trigger the graphics-processor recovery path.
    #[must_use]
    pub fn is_recoverable_graphics_processor_out_of_memory(&self) -> bool {
        let Self::RuntimeOperation { description, .. } = self else {
            return false;
        };
        description.contains("[METAL] Command buffer execution failed: Insufficient Memory")
            && description.contains("kIOGPUCommandBufferCallbackErrorOutOfMemory")
    }
}
