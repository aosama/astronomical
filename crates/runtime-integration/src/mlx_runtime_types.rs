use std::path::PathBuf;

use thiserror::Error;

/// Memory controls applied inside one MLX worker process.
///
/// These values do not alter macOS's privileged `iogpu.wired_limit_mb` sysctl.
/// The active-memory value is reused as the process residency-set limit because
/// worker startup already resolved it against the machine's system-wide ceiling.
/// The allocator-cache value independently controls reclaimable allocation reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemoryLimits {
    pub(crate) active_memory_limit_bytes: usize,
    pub(crate) allocator_cache_memory_limit_bytes: usize,
}

impl MlxMemoryLimits {
    /// Validates explicit active memory and allocator-cache byte limits.
    ///
    /// MLX treats a zero cache limit as "do not retain freed allocations".
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxMemorySnapshot {
    pub(crate) active_memory_bytes: usize,
    pub(crate) allocator_cache_memory_bytes: usize,
    pub(crate) peak_memory_bytes: usize,
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

impl MlxRuntimeError {
    /// Returns whether Metal completed a command buffer after exhausting GPU memory.
    #[must_use]
    pub fn is_recoverable_graphics_processor_out_of_memory(&self) -> bool {
        let Self::RuntimeOperation { description, .. } = self else {
            return false;
        };
        description.contains("[METAL] Command buffer execution failed: Insufficient Memory")
            && description.contains("kIOGPUCommandBufferCallbackErrorOutOfMemory")
    }
}
