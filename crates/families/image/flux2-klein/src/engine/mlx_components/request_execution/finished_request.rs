//! What a finished image request remembers for its performance report. The request's owners are
//! dropped before the report is written, so the facts the report needs are copied out first.

use runtime_integration::MlxMemorySnapshot;

use crate::dimensions::Flux2KleinImageDimensions;

/// MLX active memory, `mlx_buffer_cache` and peak memory in bytes, in that order.
pub(super) type MemoryBytes = (u64, u64, u64);

/// The same three readings as a report carries them, each absent when MLX could not be read.
pub(super) type MemoryReadings = (Option<u64>, Option<u64>, Option<u64>);

/// The request facts a performance report needs after every request owner is gone.
pub(super) struct FinishedRequest {
    pub(super) request_id: u64,
    pub(super) seed: u64,
    pub(super) dimensions: Option<Flux2KleinImageDimensions>,
    pub(super) start_memory: Option<MemoryBytes>,
    pub(super) final_memory: Option<MemoryBytes>,
}

impl FinishedRequest {
    /// The image size in pixels, or zero by zero when the request never reached its dimensions.
    pub(super) fn image_size_pixels(&self) -> (u32, u32) {
        self.dimensions.map_or((0, 0), |dimensions| {
            (dimensions.width_pixels(), dimensions.height_pixels())
        })
    }
}

pub(super) fn memory_bytes_from_snapshot(snapshot: &MlxMemorySnapshot) -> MemoryBytes {
    (
        snapshot.active_memory_bytes() as u64,
        snapshot.mlx_buffer_cache_bytes() as u64,
        snapshot.peak_memory_bytes() as u64,
    )
}

pub(super) fn memory_readings(memory: Option<MemoryBytes>) -> MemoryReadings {
    memory.map_or(
        (None, None, None),
        |(active_bytes, mlx_buffer_cache_bytes, peak_bytes)| {
            (
                Some(active_bytes),
                Some(mlx_buffer_cache_bytes),
                Some(peak_bytes),
            )
        },
    )
}

/// Cuts a failure description to 512 characters so a report never carries unbounded detail.
pub(super) fn bounded_failure_description(failure_description: Option<&str>) -> Option<String> {
    failure_description.map(|description| description.chars().take(512).collect::<String>())
}

#[cfg(test)]
#[path = "finished_request_tests.rs"]
mod finished_request_tests;
