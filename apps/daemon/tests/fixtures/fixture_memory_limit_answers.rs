//! The two answers a scripted worker gives a live memory-limit change,
//! shared by the fixture workers so each states only which answer it gives.

use ipc_protocol::{ExpertMemoryMode, WorkerEvent};

/// The worker moved to `effective_mlx_memory_ceiling_bytes`.
pub(crate) fn applied_memory_limit(effective_mlx_memory_ceiling_bytes: u64) -> WorkerEvent {
    WorkerEvent::MlxMemoryLimitChanged {
        effective_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes: 1,
        expert_memory_mode: ExpertMemoryMode::Resident,
        mlx_memory_snapshot: None,
        expert_residency: None,
    }
}

/// The worker kept its ceiling and said why.
pub(crate) fn refused_memory_limit(
    requested_mlx_memory_ceiling_bytes: u64,
    minimum_mlx_memory_ceiling_bytes: u64,
    reason: &str,
) -> WorkerEvent {
    WorkerEvent::MlxMemoryLimitRejected {
        requested_mlx_memory_ceiling_bytes,
        minimum_mlx_memory_ceiling_bytes,
        machine_mlx_memory_ceiling_bytes: 40_000_000_000,
        reason: reason.to_owned(),
    }
}
