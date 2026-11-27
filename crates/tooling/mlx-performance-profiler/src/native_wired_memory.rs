//! The native side of the wired memory readings: the C declarations of
//! `mlx_perf_read_wired_memory` and `mlx_perf_measure_wire_release_lag`,
//! and their conversion to the crate's types. Two synchronous calls into the
//! library, each writing one local struct.

use std::ffi::c_int;

use crate::measurement::ProfilerError;
use crate::native_interface::native_status;
use crate::wired_memory::{WireReleaseLag, WiredMemoryReading};

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

pub(crate) fn read_wired_memory() -> Result<WiredMemoryReading, ProfilerError> {
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

pub(crate) fn measure_wire_release_lag(
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
