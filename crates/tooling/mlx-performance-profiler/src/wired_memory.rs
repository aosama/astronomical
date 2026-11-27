//! macOS's wired memory beside MLX's own holding, and how long macOS takes
//! to un-wire what MLX releases, through the profiler library (since 0.5.0).
//!
//! On Apple silicon the memory the graphics processor works on is wired,
//! and macOS keeps wired memory under a system-wide line. An engine that
//! sizes what it keeps against "wired outside MLX" (the system's wired total
//! less MLX's active memory and buffer cache) reads that figure too high for
//! a fraction of a second after MLX releases memory, because macOS un-wires
//! it later, and too low while MLX's memory sits idle long enough to be
//! un-wired. The release lag says how long a rise must last before it is
//! someone else's memory rather than MLX's own, still wired.

use crate::measurement::ProfilerError;
#[cfg(not(target_os = "macos"))]
use crate::native_unavailable as native_library;
#[cfg(target_os = "macos")]
use crate::native_wired_memory as native_library;

/// One instant's wired memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WiredMemoryReading {
    pub system_wired_bytes: u64,
    /// The line macOS keeps wired memory under; zero when unknown.
    pub wire_limit_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_buffer_cache_bytes: u64,
    /// The system's wired total less MLX's active memory and buffer cache.
    pub wired_outside_mlx_bytes: i64,
    /// The limit less the system's wired total; zero when the limit is unknown.
    pub wire_headroom_bytes: i64,
}

/// How long macOS took to un-wire memory MLX released, on this Mac.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireReleaseLag {
    pub allocation_bytes: u64,
    pub wire_limit_bytes: u64,
    /// How far the system's wired total rose per allocation.
    pub median_wired_rise_bytes: i64,
    /// NaN when no repetition resolved.
    pub median_release_lag_microseconds: f64,
    pub largest_release_lag_microseconds: f64,
    pub repetition_count: i32,
    /// Repetitions whose un-wiring was seen in time.
    pub resolved_count: i32,
}

/// Reads the wired memory now; cheap enough to call after every evaluation.
///
/// # Errors
/// When the kernel refuses the read, or off macOS, where there is no
/// wired memory to read.
pub fn read_wired_memory() -> Result<WiredMemoryReading, ProfilerError> {
    native_library::read_wired_memory()
}

/// Allocates `allocation_bytes` (under 8 GiB) on the GPU, releases it, and
/// times how long the system's wired total takes to fall, `repetition_count`
/// times. Run it with nothing else using MLX; the library prints progress
/// with an ETA.
///
/// # Errors
/// When the arguments are refused, the probe fails, or off macOS.
pub fn measure_wire_release_lag(
    allocation_bytes: u64,
    repetition_count: i32,
) -> Result<WireReleaseLag, ProfilerError> {
    native_library::measure_wire_release_lag(allocation_bytes, repetition_count)
}
