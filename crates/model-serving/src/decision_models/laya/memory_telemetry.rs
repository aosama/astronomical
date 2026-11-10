//! What a Laya engine reports about its MLX memory.
//!
//! A Laya request is one encoder forward pass over a ModernBERT trunk, with
//! no experts, no context state and nothing kept between requests. That is
//! the embedding engine's memory shape exactly, so the ceiling is split by
//! the same owners: the resident weights, the forward workspace the process
//! has actually been seen to need, and free headroom.

use astronomical_runtime_integration::MlxMemorySnapshot;

use crate::memory::telemetry::{MlxActiveMemoryBreakdown, MlxMemoryTelemetry};
use crate::modernbert::memory_utilization::compose_modernbert_memory_ceiling_utilization;

/// Turns one allocator observation into the telemetry the worker publishes.
///
/// `resident_weight_bytes` is the size of the weight file. It overstates the
/// tensors by the file's few-kilobyte header, which is far below what the
/// allocator itself rounds, and reading it costs no parse of that header.
pub(super) fn laya_memory_telemetry(
    memory_snapshot: &MlxMemorySnapshot,
    mlx_memory_ceiling_bytes: u64,
    resident_weight_bytes: u64,
) -> MlxMemoryTelemetry {
    let active_memory_bytes = memory_snapshot.active_memory_bytes() as u64;
    let peak_memory_bytes = memory_snapshot.peak_memory_bytes() as u64;
    MlxMemoryTelemetry::new(
        active_memory_bytes,
        memory_snapshot.allocator_cache_memory_bytes() as u64,
        peak_memory_bytes,
        MlxActiveMemoryBreakdown::default(),
    )
    .with_memory_ceiling_utilization(compose_modernbert_memory_ceiling_utilization(
        mlx_memory_ceiling_bytes,
        active_memory_bytes,
        resident_weight_bytes,
        // The peak above current use is the largest forward pass seen so
        // far: evidence of the workspace the next request will need.
        peak_memory_bytes.saturating_sub(active_memory_bytes),
    ))
}
