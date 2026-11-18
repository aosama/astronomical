//! Unit contract for peer-mirrored positional-read metrics.

use std::sync::Arc;

use crate::PositionalFileReadMetrics;

#[test]
fn should_mirror_read_counters_into_the_installed_peer() {
    let session_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let request_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    request_read_metrics.install_peer_read_metrics(Arc::clone(&session_read_metrics));

    let read_succeeded = request_read_metrics.measure_read(4_096, || true);
    assert!(read_succeeded);
    request_read_metrics.measure_read(8_192, || true);

    let request_snapshot = request_read_metrics.snapshot();
    let session_snapshot = session_read_metrics.snapshot();
    assert_eq!(request_snapshot.read_call_count, 2);
    assert_eq!(session_snapshot.read_call_count, 2);
    assert_eq!(session_snapshot.read_byte_count, 12_288);
    // An empty read can finish inside one clock tick, so the mirror contract
    // is equality with the request aggregate rather than a positive duration.
    assert_eq!(
        session_snapshot.total_read_elapsed_nanoseconds,
        request_snapshot.total_read_elapsed_nanoseconds
    );
    assert_eq!(
        session_snapshot.maximum_read_elapsed_nanoseconds,
        request_snapshot.maximum_read_elapsed_nanoseconds
    );
    assert_eq!(session_snapshot.read_failure_count, 0);
}

#[test]
fn should_mirror_failed_reads_into_the_peer_as_failures() {
    let session_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    let request_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    request_read_metrics.install_peer_read_metrics(Arc::clone(&session_read_metrics));

    let read_succeeded = request_read_metrics.measure_read(4_096, || false);
    assert!(!read_succeeded);

    let session_snapshot = session_read_metrics.snapshot();
    assert_eq!(session_snapshot.read_call_count, 1);
    assert_eq!(
        session_snapshot.read_byte_count, 0,
        "failed reads add no bytes"
    );
    assert_eq!(session_snapshot.read_failure_count, 1);
}

#[test]
fn should_keep_an_unmirrored_aggregate_independent() {
    let standalone_read_metrics = PositionalFileReadMetrics::default();
    standalone_read_metrics.measure_read(512, || true);
    let standalone_snapshot = standalone_read_metrics.snapshot();
    assert_eq!(standalone_snapshot.read_call_count, 1);
    assert_eq!(standalone_snapshot.read_byte_count, 512);
}

#[test]
fn should_ignore_a_second_peer_install() {
    let first_peer_metrics = Arc::new(PositionalFileReadMetrics::default());
    let second_peer_metrics = Arc::new(PositionalFileReadMetrics::default());
    let request_read_metrics = Arc::new(PositionalFileReadMetrics::default());
    request_read_metrics.install_peer_read_metrics(Arc::clone(&first_peer_metrics));
    request_read_metrics.install_peer_read_metrics(Arc::clone(&second_peer_metrics));

    request_read_metrics.measure_read(1_024, || true);

    assert_eq!(first_peer_metrics.snapshot().read_call_count, 1);
    assert_eq!(second_peer_metrics.snapshot().read_call_count, 0);
}

#[test]
fn should_add_a_batch_read_elsewhere_to_itself_and_its_peer() {
    let session_metrics = Arc::new(PositionalFileReadMetrics::default());
    let request_metrics = PositionalFileReadMetrics::default();
    request_metrics.install_peer_read_metrics(Arc::clone(&session_metrics));
    request_metrics.record_completed_batch(10, 4_096, std::time::Duration::from_micros(800));
    request_metrics.record_completed_batch(2, 1_024, std::time::Duration::from_micros(300));
    for snapshot in [request_metrics.snapshot(), session_metrics.snapshot()] {
        assert_eq!(snapshot.read_call_count, 12);
        assert_eq!(snapshot.read_byte_count, 5_120);
        assert_eq!(snapshot.total_read_elapsed_nanoseconds, 1_100_000);
        assert_eq!(snapshot.maximum_read_elapsed_nanoseconds, 800_000);
        assert_eq!(snapshot.read_failure_count, 0);
    }
}

#[test]
fn should_record_an_empty_batch_as_nothing_read() {
    let read_metrics = PositionalFileReadMetrics::default();
    read_metrics.record_completed_batch(0, 0, std::time::Duration::ZERO);
    let snapshot = read_metrics.snapshot();
    assert_eq!(snapshot.read_call_count, 0);
    assert_eq!(snapshot.read_byte_count, 0);
    assert_eq!(snapshot.total_read_elapsed_nanoseconds, 0);
}
