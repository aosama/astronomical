//! macOS's wired memory beside MLX's own holding, and how long macOS takes
//! to un-wire what MLX releases, through the profiler library (0.5.0).
//!
//! On Apple silicon the memory the graphics processor works on is wired,
//! and macOS keeps wired memory under a system-wide line. An engine that
//! sizes what it keeps against "wired outside MLX" (the system's wired total
//! less MLX's active memory and buffer cache) reads that figure too high for
//! a fraction of a second after MLX releases memory, because macOS un-wires
//! it later, and too low while MLX's memory sits idle long enough to be
//! un-wired. The release lag says how long a rise must last before it is
//! someone else's memory rather than MLX's own, still wired.

use std::ffi::c_int;

use crate::measurement::ProfilerError;
use crate::native_interface::native_status;

/// `mlx_perf_wired_memory` of `mlx_perf_c_api.h`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeWiredMemory {
    system_wired_bytes: u64,
    wire_limit_bytes: u64,
    mlx_active_bytes: u64,
    mlx_cache_bytes: u64,
    wired_outside_mlx_bytes: i64,
    wire_headroom_bytes: i64,
}

/// `mlx_perf_wire_release_lag`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeWireReleaseLag {
    allocation_bytes: u64,
    wire_limit_bytes: u64,
    median_wired_rise_bytes: i64,
    median_release_lag_us: f64,
    largest_release_lag_us: f64,
    repetition_count: c_int,
    resolved_count: c_int,
}

unsafe extern "C" {
    fn mlx_perf_read_wired_memory(reading: *mut NativeWiredMemory) -> c_int;
    fn mlx_perf_measure_wire_release_lag(
        allocation_bytes: u64,
        repetition_count: c_int,
        result: *mut NativeWireReleaseLag,
    ) -> c_int;
}

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
/// When the kernel refuses the read.
pub fn read_wired_memory() -> Result<WiredMemoryReading, ProfilerError> {
    let mut native_reading = NativeWiredMemory::default();
    // SAFETY: the pointer is to a uniquely owned, initialized struct laid
    // out field for field as the C declaration; the call only writes it.
    let status = unsafe { mlx_perf_read_wired_memory(&raw mut native_reading) };
    native_status(status)?;
    Ok(WiredMemoryReading {
        system_wired_bytes: native_reading.system_wired_bytes,
        wire_limit_bytes: native_reading.wire_limit_bytes,
        mlx_active_bytes: native_reading.mlx_active_bytes,
        mlx_buffer_cache_bytes: native_reading.mlx_cache_bytes,
        wired_outside_mlx_bytes: native_reading.wired_outside_mlx_bytes,
        wire_headroom_bytes: native_reading.wire_headroom_bytes,
    })
}

/// Allocates `allocation_bytes` (under 8 GiB) on the GPU, releases it, and
/// times how long the system's wired total takes to fall, `repetition_count`
/// times. Run it with nothing else using MLX; the library prints progress
/// with an ETA.
///
/// # Errors
/// When the arguments are refused or the probe fails.
pub fn measure_wire_release_lag(
    allocation_bytes: u64,
    repetition_count: i32,
) -> Result<WireReleaseLag, ProfilerError> {
    let mut native_lag = NativeWireReleaseLag::default();
    // SAFETY: the pointer is to a uniquely owned, initialized struct laid
    // out field for field as the C declaration; the call only writes it.
    let status = unsafe {
        mlx_perf_measure_wire_release_lag(allocation_bytes, repetition_count, &raw mut native_lag)
    };
    native_status(status)?;
    Ok(WireReleaseLag {
        allocation_bytes: native_lag.allocation_bytes,
        wire_limit_bytes: native_lag.wire_limit_bytes,
        median_wired_rise_bytes: native_lag.median_wired_rise_bytes,
        median_release_lag_microseconds: native_lag.median_release_lag_us,
        largest_release_lag_microseconds: native_lag.largest_release_lag_us,
        repetition_count: native_lag.repetition_count,
        resolved_count: native_lag.resolved_count,
    })
}
