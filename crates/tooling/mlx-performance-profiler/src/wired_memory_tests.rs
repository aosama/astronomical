use serial_test::file_serial;

use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::wired_memory::read_wired_memory;

/// MLX creates its Metal device on first use, and only the repository's
/// runtime prepares what that needs; read before it and MLX traps.
fn started_mlx_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(8 << 30, 1 << 30).expect("the memory limits are valid"),
    )
    .expect("the MLX runtime initializes")
}

#[test]
#[file_serial]
fn should_read_a_wired_total_that_covers_what_it_says_is_outside_mlx() {
    let _mlx_runtime = started_mlx_runtime();
    let reading = read_wired_memory().expect("macOS reports its wired memory");
    assert!(reading.system_wired_bytes > 0);
    let mlx_held_bytes = reading.mlx_active_bytes + reading.mlx_buffer_cache_bytes;
    assert_eq!(
        reading.wired_outside_mlx_bytes,
        reading.system_wired_bytes as i64 - mlx_held_bytes as i64
    );
}

#[test]
#[file_serial]
fn should_give_the_room_under_the_wire_limit_when_the_limit_is_known() {
    let _mlx_runtime = started_mlx_runtime();
    let reading = read_wired_memory().expect("macOS reports its wired memory");
    if reading.wire_limit_bytes == 0 {
        assert_eq!(
            reading.wire_headroom_bytes, 0,
            "no limit, no headroom figure"
        );
    } else {
        assert_eq!(
            reading.wire_headroom_bytes,
            reading.wire_limit_bytes as i64 - reading.system_wired_bytes as i64
        );
    }
}

/// How long macOS keeps memory wired after MLX releases it, on this Mac:
/// the time a rise in "wired outside MLX" must outlast before it can be
/// another application's memory rather than MLX's own, still wired. Prints
/// the median and the largest lag over ten releases of 1 GiB.
#[test]
#[file_serial]
#[ignore = "allocates and releases 1 GiB on the GPU ten times; run with nothing else using MLX"]
fn should_measure_how_long_macos_keeps_released_mlx_memory_wired() {
    let _mlx_runtime = started_mlx_runtime();
    let release_lag = crate::wired_memory::measure_wire_release_lag(1 << 30, 10)
        .expect("the wire release probe runs");
    eprintln!(
        "[wire-release-lag] allocation_gb={:.2} median_wired_rise_gb={:.2} median_lag_ms={:.1} largest_lag_ms={:.1} resolved={}/{}",
        release_lag.allocation_bytes as f64 / f64::from(1_u32 << 30),
        release_lag.median_wired_rise_bytes as f64 / f64::from(1_u32 << 30),
        release_lag.median_release_lag_microseconds / 1_000.0,
        release_lag.largest_release_lag_microseconds / 1_000.0,
        release_lag.resolved_count,
        release_lag.repetition_count,
    );
    assert!(
        release_lag.resolved_count > 0,
        "at least one release must be seen un-wired in time"
    );
}
