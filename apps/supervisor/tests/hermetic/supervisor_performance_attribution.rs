use std::{
    io::{self, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use astronomical_supervisor::{
    SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
    SupervisorPerformanceOperation,
};
use tokio::sync::Notify;

const TEST_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[tokio::test]
async fn should_execute_disabled_async_measurement_without_reading_the_clock() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let clock_call_count = Arc::new(AtomicUsize::new(0));
        let clock_call_count_for_clock = Arc::clone(&clock_call_count);
        let attribution_log =
            SupervisorPerformanceAttributionLog::from_writer_and_clock_when_enabled(
                io::sink(),
                move || {
                    clock_call_count_for_clock.fetch_add(1, Ordering::SeqCst);
                    Ok(1_000)
                },
                false,
            );

        let operation_output = attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::ManifestFetch,
                || async { 42_u64 },
                |_| SupervisorPerformanceMeasurement::success(),
            )
            .await
            .expect("disabled attribution must not affect async work");

        assert_eq!(operation_output, 42);
        assert_eq!(clock_call_count.load(Ordering::SeqCst), 0);
    })
    .await
    .expect("disabled async attribution coverage must remain bounded");
}

#[tokio::test]
async fn should_record_deterministic_async_manifest_measurement() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let written_bytes = Arc::new(Mutex::new(Vec::new()));
        let clock_call_count = Arc::new(AtomicUsize::new(0));
        let clock_call_count_for_clock = Arc::clone(&clock_call_count);
        let attribution_log = SupervisorPerformanceAttributionLog::from_writer_and_clock(
            SharedWriter::new(Arc::clone(&written_bytes)),
            move || {
                let clock_index = clock_call_count_for_clock.fetch_add(1, Ordering::SeqCst);
                Ok([1_000, 1_025][clock_index])
            },
        );
        let measurement = SupervisorPerformanceMeasurement::success()
            .with_manifest_fetch(
                "astronomical-test/example-qwen",
                TEST_REVISION,
                3,
                4_000_000_000,
            )
            .expect("the fictional manifest metadata must be valid");

        let operation_output = attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::ManifestFetch,
                || async { "manifest fetched" },
                |_| measurement,
            )
            .await
            .expect("enabled attribution must record completed async work");

        assert_eq!(operation_output, "manifest fetched");
        let attribution_record = only_record(&written_bytes);
        assert_eq!(attribution_record["operation"], "manifest_fetch");
        assert_eq!(attribution_record["started_at_unix_millis"], 1_000);
        assert_eq!(attribution_record["ended_at_unix_millis"], 1_025);
        assert_eq!(attribution_record["outcome"], "success");
        assert_eq!(
            attribution_record["huggingface_id"],
            "astronomical-test/example-qwen"
        );
        assert_eq!(attribution_record["revision"], TEST_REVISION);
        assert_eq!(attribution_record["manifest_file_count"], 3);
        assert_eq!(
            attribution_record["manifest_total_bytes"],
            4_000_000_000_u64
        );
    })
    .await
    .expect("enabled async attribution coverage must remain bounded");
}

#[tokio::test]
async fn should_not_hold_the_writer_lock_while_an_async_operation_is_blocked() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let written_bytes = Arc::new(Mutex::new(Vec::new()));
        let clock_value = Arc::new(AtomicUsize::new(1_000));
        let clock_value_for_clock = Arc::clone(&clock_value);
        let attribution_log = SupervisorPerformanceAttributionLog::from_writer_and_clock(
            SharedWriter::new(Arc::clone(&written_bytes)),
            move || Ok(clock_value_for_clock.fetch_add(1, Ordering::SeqCst) as u64),
        );
        let first_operation_started = Arc::new(Notify::new());
        let release_first_operation = Arc::new(Notify::new());
        let first_log = attribution_log.clone();
        let first_started_signal = Arc::clone(&first_operation_started);
        let first_release_signal = Arc::clone(&release_first_operation);

        let first_operation = tokio::spawn(async move {
            first_log
                .measure_async_operation(
                    SupervisorPerformanceOperation::FileTransfer,
                    || async move {
                        first_started_signal.notify_one();
                        first_release_signal.notified().await;
                    },
                    |_| file_transfer_measurement(),
                )
                .await
        });
        first_operation_started.notified().await;

        attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::Verification,
                || async {},
                |_| verification_measurement(),
            )
            .await
            .expect("the second operation must record while the first operation is awaiting");
        assert_eq!(record_count(&written_bytes), 1);

        release_first_operation.notify_one();
        first_operation
            .await
            .expect("the first task must remain joinable")
            .expect("the first operation must record after release");
        assert_eq!(record_count(&written_bytes), 2);
    })
    .await
    .expect("concurrent async attribution coverage must remain bounded");
}

#[tokio::test]
async fn should_report_end_clock_failure_only_after_async_operation_completes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let operation_completed = Arc::new(AtomicBool::new(false));
        let operation_completed_for_operation = Arc::clone(&operation_completed);
        let clock_call_count = Arc::new(AtomicUsize::new(0));
        let clock_call_count_for_clock = Arc::clone(&clock_call_count);
        let attribution_log =
            SupervisorPerformanceAttributionLog::from_writer_and_clock(io::sink(), move || {
                if clock_call_count_for_clock.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(1_000)
                } else {
                    assert!(operation_completed.load(Ordering::SeqCst));
                    Err(io::Error::other("intentional end clock failure"))
                }
            });

        let attribution_error = attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::DiskPreflight,
                || async move {
                    operation_completed_for_operation.store(true, Ordering::SeqCst);
                },
                |_| disk_preflight_measurement(),
            )
            .await
            .expect_err("end clock failure must remain typed");

        assert!(
            attribution_error
                .to_string()
                .contains("intentional end clock failure")
        );
    })
    .await
    .expect("end clock failure coverage must remain bounded");
}

#[tokio::test]
async fn should_report_write_failure_only_after_async_operation_completes() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let operation_completed = Arc::new(AtomicBool::new(false));
        let operation_completed_for_operation = Arc::clone(&operation_completed);
        let attribution_log = SupervisorPerformanceAttributionLog::from_writer_and_clock(
            CompletionCheckingWriter {
                operation_completed: Arc::clone(&operation_completed),
            },
            deterministic_clock(),
        );

        let attribution_error = attribution_log
            .measure_async_operation(
                SupervisorPerformanceOperation::Verification,
                || async move {
                    operation_completed_for_operation.store(true, Ordering::SeqCst);
                },
                |_| verification_measurement(),
            )
            .await
            .expect_err("write failure must remain typed");

        assert!(
            attribution_error
                .to_string()
                .contains("intentional write failure")
        );
    })
    .await
    .expect("write failure coverage must remain bounded");
}

#[test]
fn should_reject_noncanonical_file_transfer_attribution_paths() {
    for invalid_relative_path in [
        "weights//model.safetensors",
        "weights/./model.safetensors",
        "weights/model.safetensors/",
    ] {
        let measurement_error = SupervisorPerformanceMeasurement::success()
            .with_file_transfer(
                "astronomical-test/example-qwen",
                TEST_REVISION,
                invalid_relative_path,
                0,
                1,
            )
            .expect_err("attribution must use the canonical durable file identity");

        assert_eq!(measurement_error.kind(), io::ErrorKind::InvalidInput);
    }
}

fn deterministic_clock() -> impl Fn() -> io::Result<u64> + Send + Sync + 'static {
    let clock_call_count = AtomicUsize::new(0);
    move || Ok(1_000 + clock_call_count.fetch_add(1, Ordering::SeqCst) as u64)
}

fn disk_preflight_measurement() -> SupervisorPerformanceMeasurement {
    SupervisorPerformanceMeasurement::failure()
        .with_disk_preflight("astronomical-test/example-qwen", TEST_REVISION, 100, 50)
        .expect("disk preflight attribution fixture should be valid")
}

fn file_transfer_measurement() -> SupervisorPerformanceMeasurement {
    SupervisorPerformanceMeasurement::success()
        .with_file_transfer(
            "astronomical-test/example-qwen",
            TEST_REVISION,
            "model.safetensors",
            0,
            100,
        )
        .expect("file transfer attribution fixture should be valid")
}

fn verification_measurement() -> SupervisorPerformanceMeasurement {
    SupervisorPerformanceMeasurement::cancelled()
        .with_verification("astronomical-test/example-qwen", TEST_REVISION, 1, 100)
        .expect("verification attribution fixture should be valid")
}

fn only_record(written_bytes: &Arc<Mutex<Vec<u8>>>) -> serde_json::Value {
    let written_bytes = written_bytes
        .lock()
        .expect("test writer must remain available");
    serde_json::from_slice(written_bytes.as_slice()).expect("one JSON record must be written")
}

fn record_count(written_bytes: &Arc<Mutex<Vec<u8>>>) -> usize {
    let written_bytes = written_bytes
        .lock()
        .expect("test writer must remain available");
    written_bytes
        .split(|byte| *byte == b'\n')
        .filter(|record| !record.is_empty())
        .count()
}

struct SharedWriter {
    written_bytes: Arc<Mutex<Vec<u8>>>,
}

impl SharedWriter {
    fn new(written_bytes: Arc<Mutex<Vec<u8>>>) -> Self {
        Self { written_bytes }
    }
}

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.written_bytes
            .lock()
            .map_err(|_| io::Error::other("test writer lock was poisoned"))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct CompletionCheckingWriter {
    operation_completed: Arc<AtomicBool>,
}

impl Write for CompletionCheckingWriter {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        assert!(self.operation_completed.load(Ordering::SeqCst));
        Err(io::Error::other("intentional write failure"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("intentional write failure"))
    }
}
