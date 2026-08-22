//! Acceptance journey for ordered disk admission, manifest discovery, and durable exact state.

use std::{
    collections::VecDeque,
    io::{self, Write},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use astronomical_supervisor::{
    DiskCapacityQuery, DownloadCatalog, DownloadDiskPreflight, DownloadDiskPreflightError,
    DownloadJobStore, DownloadManifestPreflight, DownloadManifestPreflightError, HubHttpRequest,
    HubHttpResponse, HubTransport, HubTransportError, HubTransportFuture, HuggingFaceHub,
    SupervisorPerformanceAttributionLog,
};
use tempfile::TempDir;

const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const GIT_BLOB_SHA1: &str = "1111111111111111111111111111111111111111";

#[tokio::test]
async fn should_check_disk_before_hub_and_persist_exact_manifest_after_the_second_check() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = DownloadJobStore::new(test_directory.path().join("models"));
        let capacity_query_count = Arc::new(AtomicUsize::new(0));
        let disk_preflight = DownloadDiskPreflight::new(SequenceCapacityQuery {
            query_count: Arc::clone(&capacity_query_count),
            available_bytes: Arc::new(Mutex::new([1_000, 1_000].into())),
        });
        let transport = Arc::new(OrderingTransport {
            capacity_query_count: Arc::clone(&capacity_query_count),
            request_count: AtomicUsize::new(0),
            responses: Mutex::new(
                [
                    json_response(serde_json::json!({
                        "id": REPOSITORY_ID,
                        "sha": REVISION,
                        "private": false,
                        "gated": false
                    })),
                    json_response(serde_json::json!([
                        {"type":"file","size":9,"path":"config.json","oid":GIT_BLOB_SHA1}
                    ])),
                ]
                .into(),
            ),
        });
        let written_attribution = Arc::new(Mutex::new(Vec::new()));
        let clock_call_count = Arc::new(AtomicUsize::new(0));
        let clock_call_count_for_clock = Arc::clone(&clock_call_count);
        let attribution_log = SupervisorPerformanceAttributionLog::from_writer_and_clock(
            SharedWriter(Arc::clone(&written_attribution)),
            move || Ok(1_000 + clock_call_count_for_clock.fetch_add(1, Ordering::SeqCst) as u64),
        );
        let catalog = DownloadCatalog::parse_json(&catalog_json())
            .expect("fictional catalog should be valid");
        let manifest_preflight = DownloadManifestPreflight::new(
            job_store.clone(),
            disk_preflight,
            HuggingFaceHub::new(transport.clone()),
            attribution_log,
        );

        let exact_job = manifest_preflight
            .prepare(&catalog.entries()[0], 100, 200, 300)
            .await
            .expect("the ordered preflight and manifest journey should complete");

        assert_eq!(capacity_query_count.load(Ordering::SeqCst), 2);
        assert_eq!(exact_job.bytes_total(), 9);
        assert!(exact_job.has_exact_manifest());
        assert_eq!(
            job_store
                .load()
                .expect("exact durable job should load")
                .expect("exact durable job should exist"),
            exact_job
        );
        assert_eq!(transport.remaining_response_count(), 0);
        assert_eq!(transport.request_count.load(Ordering::SeqCst), 2);
        let operations = attribution_operations(&written_attribution);
        assert_eq!(
            operations,
            ["disk_preflight", "manifest_fetch", "disk_preflight"]
        );
    })
    .await
    .expect("manifest preflight acceptance journey should remain bounded");
}

#[tokio::test]
async fn should_stop_before_hub_io_when_initial_disk_admission_fails() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = DownloadJobStore::new(test_directory.path().join("models"));
        let capacity_query_count = Arc::new(AtomicUsize::new(0));
        let transport = Arc::new(OrderingTransport {
            capacity_query_count: Arc::clone(&capacity_query_count),
            request_count: AtomicUsize::new(0),
            responses: Mutex::new(VecDeque::new()),
        });
        let catalog = DownloadCatalog::parse_json(&catalog_json())
            .expect("fictional catalog should be valid");
        let manifest_preflight = DownloadManifestPreflight::new(
            job_store.clone(),
            DownloadDiskPreflight::new(SequenceCapacityQuery {
                query_count: Arc::clone(&capacity_query_count),
                available_bytes: Arc::new(Mutex::new([100].into())),
            }),
            HuggingFaceHub::new(transport.clone()),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        );

        let preflight_error = manifest_preflight
            .prepare(&catalog.entries()[0], 100, 200, 300)
            .await
            .expect_err("insufficient initial capacity must stop the journey");

        assert!(matches!(
            preflight_error,
            DownloadManifestPreflightError::Disk(DownloadDiskPreflightError::InsufficientSpace {
                required_bytes: 101,
                available_bytes: 100,
            })
        ));
        assert_eq!(capacity_query_count.load(Ordering::SeqCst), 1);
        assert_eq!(transport.request_count.load(Ordering::SeqCst), 0);

        let retry_capacity_count = Arc::new(AtomicUsize::new(0));
        let retry_transport = Arc::new(OrderingTransport {
            capacity_query_count: Arc::clone(&retry_capacity_count),
            request_count: AtomicUsize::new(0),
            responses: Mutex::new(
                [
                    json_response(serde_json::json!({
                        "id": REPOSITORY_ID,
                        "sha": REVISION,
                        "private": false,
                        "gated": false
                    })),
                    json_response(serde_json::json!([
                        {"type":"file","size":9,"path":"config.json","oid":GIT_BLOB_SHA1}
                    ])),
                ]
                .into(),
            ),
        });
        let retry_preflight = DownloadManifestPreflight::new(
            job_store,
            DownloadDiskPreflight::new(SequenceCapacityQuery {
                query_count: retry_capacity_count,
                available_bytes: Arc::new(Mutex::new([1_000, 1_000].into())),
            }),
            HuggingFaceHub::new(retry_transport.clone()),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        );

        let retried_job = retry_preflight
            .prepare(&catalog.entries()[0], 400, 500, 600)
            .await
            .expect("a paused premanifest job should retry without destructive cancellation");

        assert!(retried_job.has_exact_manifest());
        assert_eq!(retry_transport.request_count.load(Ordering::SeqCst), 2);
    })
    .await
    .expect("failed initial disk admission should remain bounded");
}

#[tokio::test]
async fn should_resume_persisted_exact_manifest_with_synchronized_staging_progress() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let job_store = DownloadJobStore::new(test_directory.path().join("models"));
        let catalog = DownloadCatalog::parse_json(&catalog_json())
            .expect("fictional catalog should be valid");
        let first_capacity_count = Arc::new(AtomicUsize::new(0));
        let first_transport = Arc::new(OrderingTransport {
            capacity_query_count: Arc::clone(&first_capacity_count),
            request_count: AtomicUsize::new(0),
            responses: Mutex::new(
                [
                    json_response(serde_json::json!({
                        "id": REPOSITORY_ID,
                        "sha": REVISION,
                        "private": false,
                        "gated": false
                    })),
                    json_response(serde_json::json!([
                        {"type":"file","size":9,"path":"config.json","oid":GIT_BLOB_SHA1}
                    ])),
                ]
                .into(),
            ),
        });
        let first_preflight = DownloadManifestPreflight::new(
            job_store.clone(),
            DownloadDiskPreflight::new(SequenceCapacityQuery {
                query_count: first_capacity_count,
                available_bytes: Arc::new(Mutex::new([1_000, 5].into())),
            }),
            HuggingFaceHub::new(first_transport),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        );

        assert!(matches!(
            first_preflight
                .prepare(&catalog.entries()[0], 100, 200, 300)
                .await,
            Err(DownloadManifestPreflightError::Disk(
                DownloadDiskPreflightError::InsufficientSpace {
                    required_bytes: 9,
                    available_bytes: 5,
                }
            ))
        ));
        let staged_file = test_directory
            .path()
            .join("models/.incomplete/astronomical-test/example-qwen/config.json");
        std::fs::create_dir_all(
            staged_file
                .parent()
                .expect("staged file should have a parent"),
        )
        .expect("staging directory should be created");
        std::fs::write(&staged_file, b"Rome").expect("staged progress should be written");
        let resumed_capacity_count = Arc::new(AtomicUsize::new(0));
        let resumed_transport = Arc::new(OrderingTransport {
            capacity_query_count: Arc::clone(&resumed_capacity_count),
            request_count: AtomicUsize::new(0),
            responses: Mutex::new(VecDeque::new()),
        });
        let resumed_preflight = DownloadManifestPreflight::new(
            job_store,
            DownloadDiskPreflight::new(SequenceCapacityQuery {
                query_count: resumed_capacity_count,
                available_bytes: Arc::new(Mutex::new([5].into())),
            }),
            HuggingFaceHub::new(resumed_transport.clone()),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        );

        let resumed_job = resumed_preflight
            .prepare(&catalog.entries()[0], 400, 500, 600)
            .await
            .expect("exact manifest should resume without another Hub request");

        assert_eq!(resumed_job.bytes_completed(), 4);
        assert_eq!(resumed_job.remaining_bytes(), 5);
        assert_eq!(resumed_transport.request_count.load(Ordering::SeqCst), 0);
    })
    .await
    .expect("exact-manifest resume journey should remain bounded");
}

#[derive(Clone)]
struct SequenceCapacityQuery {
    query_count: Arc<AtomicUsize>,
    available_bytes: Arc<Mutex<VecDeque<u64>>>,
}

impl DiskCapacityQuery for SequenceCapacityQuery {
    fn available_space_bytes(&self, existing_same_volume_path: &Path) -> io::Result<u64> {
        assert!(existing_same_volume_path.exists());
        self.query_count.fetch_add(1, Ordering::SeqCst);
        self.available_bytes
            .lock()
            .map_err(|_| io::Error::other("capacity sequence lock was poisoned"))?
            .pop_front()
            .ok_or_else(|| io::Error::other("capacity sequence was exhausted"))
    }
}

struct OrderingTransport {
    capacity_query_count: Arc<AtomicUsize>,
    request_count: AtomicUsize,
    responses: Mutex<VecDeque<HubHttpResponse>>,
}

impl OrderingTransport {
    fn remaining_response_count(&self) -> usize {
        self.responses
            .lock()
            .expect("transport response lock should remain available")
            .len()
    }
}

impl HubTransport for OrderingTransport {
    fn execute(&self, _request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            if self.capacity_query_count.load(Ordering::SeqCst) == 0 {
                return Err(HubTransportError::new(
                    "Hub I/O started before initial disk admission",
                ));
            }
            self.request_count.fetch_add(1, Ordering::SeqCst);
            self.responses
                .lock()
                .map_err(|_| HubTransportError::new("transport response lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected Hub request"))
        })
    }
}

fn json_response(body: serde_json::Value) -> HubHttpResponse {
    HubHttpResponse::try_new(
        200,
        [],
        [serde_json::to_vec(&body).expect("scripted Hub body should serialize")],
    )
    .expect("scripted Hub response should remain bounded")
}

fn catalog_json() -> String {
    format!(
        "{{\"schema_version\":1,\"entries\":[{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"display_name\":\"Example model\",\"family\":\"qwen3_5\",\"approximate_size_bytes\":100,\"public\":true}}]}}"
    )
}

struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("attribution output lock was poisoned"))?
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn attribution_operations(written_attribution: &Arc<Mutex<Vec<u8>>>) -> Vec<String> {
    let written_attribution = written_attribution
        .lock()
        .expect("attribution output lock should remain available");
    written_attribution
        .split(|byte| *byte == b'\n')
        .filter(|record| !record.is_empty())
        .map(|record| {
            serde_json::from_slice::<serde_json::Value>(record)
                .expect("attribution record should be JSON")["operation"]
                .as_str()
                .expect("attribution operation should be text")
                .to_owned()
        })
        .collect()
}
