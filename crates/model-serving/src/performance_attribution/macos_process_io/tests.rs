//! Pure monotonic-delta arithmetic for macOS process I/O snapshots.

use super::{MacosProcessIoError, MacosProcessIoSnapshot};
#[cfg(target_os = "macos")]
use super::{sample_current_process_io, sample_process_io_for_process_id_on_platform};

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

/// Bytes past the end of `libc::rusage_info_v4` that the canary test fills;
/// one u64 is the size of the overrun that once corrupted a saved register.
#[cfg(target_os = "macos")]
const CANARY_TAIL_BYTE_COUNT: usize = 64;
#[cfg(target_os = "macos")]
const CANARY_BYTE: u8 = 0xA5;

/// The kernel writes the V4 flavor's full structure; a Rust layout shorter
/// than that lets the syscall overwrite whatever follows it on the stack.
#[cfg(target_os = "macos")]
#[test]
fn should_keep_the_v4_flavor_write_inside_libcs_rusage_info_v4() {
    let structure_byte_count = std::mem::size_of::<libc::rusage_info_v4>();
    let mut canary_guarded_buffer =
        vec![CANARY_BYTE; structure_byte_count + CANARY_TAIL_BYTE_COUNT];
    let current_process_id = i32::try_from(std::process::id()).expect("pid fits c_int");

    // SAFETY: the buffer is larger than the structure the flavor writes and
    // uniquely borrowed for this synchronous call.
    let status = unsafe {
        libc::proc_pid_rusage(
            current_process_id,
            libc::RUSAGE_INFO_V4,
            canary_guarded_buffer.as_mut_ptr().cast(),
        )
    };

    assert_eq!(status, 0);
    let canary_tail = &canary_guarded_buffer[structure_byte_count..];
    assert!(
        canary_tail.iter().all(is_canary_byte),
        "kernel wrote past rusage_info_v4"
    );
}

#[cfg(target_os = "macos")]
fn is_canary_byte(tail_byte: &u8) -> bool {
    *tail_byte == CANARY_BYTE
}

#[cfg(target_os = "macos")]
#[test]
fn should_sample_the_current_process_twice_with_monotonic_counters() {
    let first_snapshot = sample_current_process_io().expect("first sample");
    let second_snapshot = sample_current_process_io().expect("second sample");

    assert!(second_snapshot.delta_since(first_snapshot).is_ok());
}

#[cfg(target_os = "macos")]
#[test]
fn should_reject_a_process_id_wider_than_c_int() {
    assert!(matches!(
        sample_process_io_for_process_id_on_platform(u32::MAX),
        Err(MacosProcessIoError::SamplingFailed { os_error_code }) if os_error_code == libc::EINVAL
    ));
}

#[cfg(target_os = "macos")]
#[test]
fn should_report_a_sampling_failure_for_a_process_that_does_not_exist() {
    let nonexistent_process_id = 99_999_999;

    assert!(matches!(
        sample_process_io_for_process_id_on_platform(nonexistent_process_id),
        Err(MacosProcessIoError::SamplingFailed { .. })
    ));
}
