use astronomical_runtime_integration::{MacosProcessIoError, MacosProcessIoSnapshot};

#[test]
fn should_calculate_monotonic_process_io_deltas() {
    let earlier_snapshot = MacosProcessIoSnapshot::from_cumulative_bytes(1_000, 400);
    let later_snapshot = MacosProcessIoSnapshot::from_cumulative_bytes(1_750, 460);

    let process_io_delta = later_snapshot
        .delta_since(earlier_snapshot)
        .expect("monotonic process I/O counters should produce a delta");

    assert_eq!(process_io_delta.physical_disk_read_bytes(), 750);
    assert_eq!(process_io_delta.physical_disk_written_bytes(), 60);
}

#[test]
fn should_reject_a_regressed_process_io_counter() {
    let earlier_snapshot = MacosProcessIoSnapshot::from_cumulative_bytes(1_000, 400);
    let later_snapshot = MacosProcessIoSnapshot::from_cumulative_bytes(999, 460);

    let process_io_error = later_snapshot
        .delta_since(earlier_snapshot)
        .expect_err("a regressed process I/O counter must not wrap");

    assert_eq!(
        process_io_error,
        MacosProcessIoError::CounterRegressed {
            counter_name: "ri_diskio_bytesread",
            earlier_bytes: 1_000,
            later_bytes: 999,
        }
    );
}

#[cfg(target_os = "macos")]
#[test]
fn should_sample_current_macos_process_io() {
    let process_io_snapshot = astronomical_runtime_integration::sample_current_process_io()
        .expect("the current macOS process should expose resource usage");

    let unchanged_delta = process_io_snapshot
        .delta_since(process_io_snapshot)
        .expect("one snapshot compared with itself should remain monotonic");
    assert_eq!(unchanged_delta.physical_disk_read_bytes(), 0);
    assert_eq!(unchanged_delta.physical_disk_written_bytes(), 0);
}
