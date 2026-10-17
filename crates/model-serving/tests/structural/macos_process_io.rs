//! macOS-gated sampling contracts that touch real process state; the pure
//! delta-math contracts live beside the owner in src/performance_attribution.

use astronomical_model_serving::MacosProcessIoError;

#[cfg(target_os = "macos")]
#[test]
fn should_sample_current_macos_process_io() {
    let process_io_snapshot = astronomical_model_serving::sample_current_process_io()
        .expect("the current macOS process should expose resource usage");

    let unchanged_delta = process_io_snapshot
        .delta_since(process_io_snapshot)
        .expect("one snapshot compared with itself should remain monotonic");
    assert_eq!(unchanged_delta.physical_disk_read_bytes(), 0);
    assert_eq!(unchanged_delta.physical_disk_written_bytes(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn should_sample_another_process_id_with_the_same_cumulative_counters() {
    let self_process_id = std::process::id();

    let self_sampled = astronomical_model_serving::sample_current_process_io()
        .expect("self-sampling should succeed on macOS");
    let externally_sampled =
        astronomical_model_serving::sample_process_io_for_process_id(self_process_id)
            .expect("sampling this same process through its id should succeed");

    let equivalence_delta = externally_sampled
        .delta_since(self_sampled)
        .expect("two samples of one process taken back to back should be monotonic");
    // Back-to-back samples of an idle test process move by nothing observable;
    // the assertion is about API equivalence, not about zero disk activity.
    assert!(equivalence_delta.physical_disk_read_bytes() < 1024 * 1024);
}

#[cfg(target_os = "macos")]
#[test]
fn should_fail_typed_when_sampling_a_process_id_that_does_not_exist() {
    // Process id 0 is reserved for the kernel pager and is never a sampleable
    // user process; macOS reports it as an invalid identifier.
    let sampling_error = astronomical_model_serving::sample_process_io_for_process_id(0)
        .expect_err("the reserved kernel process id must not be sampleable");

    assert!(matches!(
        sampling_error,
        MacosProcessIoError::SamplingFailed { .. }
    ));
}
