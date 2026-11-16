//! The daemon's attribution records around loading the catalog: absent when
//! attribution is off, one success or failure record each when it is on, and a
//! typed error when the record cannot be written.

use std::{
    fs, io,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
    DownloadCatalog,
};

#[test]
fn should_leave_the_daemon_attribution_file_absent_when_disabled() {
    let log_directory = tempfile::tempdir().expect("a log directory should be created");
    let attribution_log = DaemonPerformanceAttributionLog::open(log_directory.path(), false)
        .expect("disabled attribution should remain inert");

    let measured_catalog = attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":5,"entries":[]}"#),
            |catalog_outcome| match catalog_outcome {
                Ok(download_catalog) => DaemonPerformanceMeasurement::successful_catalog_load(
                    download_catalog.entry_count(),
                ),
                Err(_) => DaemonPerformanceMeasurement::failure(),
            },
        )
        .expect("disabled attribution should not affect the operation")
        .expect("the measured catalog should parse");

    assert_eq!(measured_catalog.entry_count(), 0);
    assert!(
        !log_directory
            .path()
            .join("daemon-performance-attribution.jsonl")
            .exists()
    );
}

#[test]
fn should_record_success_and_failure_catalog_load_boundaries_when_enabled() {
    let log_directory = tempfile::tempdir().expect("a log directory should be created");
    let attribution_log = DaemonPerformanceAttributionLog::open(log_directory.path(), true)
        .expect("enabled attribution should open its writer");

    let successful_catalog = attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":5,"entries":[]}"#),
            |catalog_outcome| match catalog_outcome {
                Ok(download_catalog) => DaemonPerformanceMeasurement::successful_catalog_load(
                    download_catalog.entry_count(),
                ),
                Err(_) => DaemonPerformanceMeasurement::failure(),
            },
        )
        .expect("successful attribution should be written");
    assert!(successful_catalog.is_ok());

    let failed_catalog = attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":1,"entries":[]}"#),
            |_| DaemonPerformanceMeasurement::failure(),
        )
        .expect("failed operation attribution should still be written");
    assert!(failed_catalog.is_err());

    let attribution_text = fs::read_to_string(
        log_directory
            .path()
            .join("daemon-performance-attribution.jsonl"),
    )
    .expect("the attribution log should remain readable");
    let attribution_records = attribution_text
        .lines()
        .map(|json_line| {
            serde_json::from_str::<serde_json::Value>(json_line)
                .expect("each attribution row should contain JSON")
        })
        .collect::<Vec<_>>();

    assert_eq!(attribution_records.len(), 2);
    assert_eq!(attribution_records[0]["operation"], "library_catalog_load");
    assert_eq!(attribution_records[0]["outcome"], "success");
    assert_eq!(attribution_records[0]["catalog_entry_count"], 0);
    assert_eq!(attribution_records[1]["outcome"], "failure");
    for attribution_record in attribution_records {
        let started_at = attribution_record["started_at_unix_millis"]
            .as_u64()
            .expect("the start timestamp should be unsigned");
        let ended_at = attribution_record["ended_at_unix_millis"]
            .as_u64()
            .expect("the end timestamp should be unsigned");
        assert!(ended_at >= started_at);
        assert!(attribution_record["elapsed_nanoseconds"].is_u64());
    }
}

#[test]
fn should_return_a_typed_io_error_when_an_enabled_attribution_write_fails() {
    let attribution_log = DaemonPerformanceAttributionLog::from_writer(AlwaysFailingWriter);

    let attribution_error = attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            || DownloadCatalog::parse_json(r#"{"schema_version":5,"entries":[]}"#),
            |_| DaemonPerformanceMeasurement::successful_catalog_load(0),
        )
        .expect_err("required attribution write failures must remain typed");

    assert_eq!(attribution_error.kind(), io::ErrorKind::Other);
    assert!(
        attribution_error
            .to_string()
            .contains("intentional attribution failure")
    );
}

#[test]
fn should_return_a_typed_io_error_when_an_enabled_attribution_timestamp_fails() {
    let clock_call_count = Arc::new(AtomicUsize::new(0));
    let measured_operation_executed = Arc::new(AtomicBool::new(false));
    let clock_call_count_for_writer = Arc::clone(&clock_call_count);
    let measured_operation_executed_for_operation = Arc::clone(&measured_operation_executed);
    let attribution_log =
        DaemonPerformanceAttributionLog::from_writer_and_clock(io::sink(), move || {
            if clock_call_count_for_writer.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(1_000)
            } else {
                Err(io::Error::other("intentional attribution clock failure"))
            }
        });

    let attribution_error = attribution_log
        .measure_operation(
            DaemonPerformanceOperation::LibraryCatalogLoad,
            || {
                measured_operation_executed_for_operation.store(true, Ordering::SeqCst);
                DownloadCatalog::parse_json(r#"{"schema_version":5,"entries":[]}"#)
            },
            |_| DaemonPerformanceMeasurement::successful_catalog_load(0),
        )
        .expect_err("required attribution timestamp failures must remain typed");

    assert!(measured_operation_executed.load(Ordering::SeqCst));
    assert_eq!(clock_call_count.load(Ordering::SeqCst), 2);
    assert_eq!(attribution_error.kind(), io::ErrorKind::Other);
    assert!(
        attribution_error
            .to_string()
            .contains("intentional attribution clock failure")
    );
}

struct AlwaysFailingWriter;

impl Write for AlwaysFailingWriter {
    fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("intentional attribution failure"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("intentional attribution failure"))
    }
}
