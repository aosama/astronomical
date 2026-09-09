//! Acceptance journey: unexecutable catalog artifacts must fail before payload transfer.
//!
//! The user journey is "click Download on a Library card and either get a servable model or a
//! fast, honest failure". An artifact that executable discovery would never advertise used to
//! cost a complete multi-gigabyte transfer before publication failed; these journeys prove the
//! failure now happens in preflight with zero payload bytes transferred.

use std::{
    collections::VecDeque,
    io,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use astronomical_supervisor::{
    DiskCapacityQuery, DownloadCatalog, DownloadJobPublicErrorCode, DownloadJobState,
    DownloadPublicationRefresh, HubHttpRequest, HubHttpResponse, HubPayloadFuture,
    HubPayloadRequest, HubPayloadTransport, HubTransport, HubTransportError, HubTransportFuture,
    LibraryDownloadCoordinator, SupervisorPerformanceAttributionLog,
};
use tempfile::TempDir;

const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const GIT_BLOB_SHA1: &str = "1111111111111111111111111111111111111111";
const QWEN_SHARD_FILE: &str = "model-00001-of-00001.safetensors";

#[tokio::test]
async fn should_reject_an_unknown_model_type_before_transferring_any_payload() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let transport = Arc::new(MetadataCountingTransport::new([
            repository_metadata_response(),
            qwen_tree_response(),
            json_response(serde_json::json!({
                "model_type": "mystery_architecture"
            })),
        ]));
        let journey = DownloadJourney::new(transport.clone()).await;

        journey
            .start_download_and_await_failure()
            .await
            .expect_public_error(DownloadJobPublicErrorCode::ModelNotExecutable);

        assert_eq!(
            transport.metadata_request_count(),
            3,
            "preflight must stop after reading config.json metadata"
        );
        assert_eq!(
            transport.payload_request_count(),
            0,
            "no payload byte may be requested for an unexecutable artifact"
        );
    })
    .await
    .expect("unknown model family journey should remain bounded");
}

#[tokio::test]
async fn should_reject_a_shard_index_requiring_an_unselected_shard() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let transport = Arc::new(MetadataCountingTransport::new([
            repository_metadata_response(),
            qwen_tree_response(),
            servable_qwen_config_response(),
            json_response(serde_json::json!({
                "weight_map": {"model.layers.0.weight": "model-00002-of-00001.safetensors"}
            })),
        ]));
        let journey = DownloadJourney::new(transport.clone()).await;

        journey
            .start_download_and_await_failure()
            .await
            .expect_public_error(DownloadJobPublicErrorCode::ModelNotExecutable);

        assert_eq!(
            transport.metadata_request_count(),
            4,
            "the gate must read the shard index before rejecting the artifact"
        );
        assert_eq!(
            transport.payload_request_count(),
            0,
            "a mandatory missing shard must fail before any payload transfer"
        );
    })
    .await
    .expect("shard inventory journey should remain bounded");
}

#[tokio::test]
async fn should_classify_a_pipeline_artifact_by_its_index_before_its_config() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // Diffusers pipelines carry both documents; disk discovery classifies model_index.json
        // first, so the gate must too, or a pipeline whose config.json is not a weight-family
        // document would be rejected even though disk discovery would serve it.
        let transport = Arc::new(MetadataCountingTransport::new([
            repository_metadata_response(),
            flux_pipeline_tree_response(),
            flux_pipeline_index_response(),
        ]));
        let journey = DownloadJourney::new(transport.clone()).await;

        journey.start_download_and_await_transfer_attempt().await;

        assert_eq!(
            transport.metadata_request_count(),
            3,
            "the gate must classify from model_index.json and never read config.json"
        );
        assert!(
            transport.payload_request_count() >= 1,
            "an executable pipeline must pass the gate and reach payload transfer"
        );
    })
    .await
    .expect("pipeline precedence journey should remain bounded");
}

#[tokio::test]
async fn should_surface_a_gate_retrieval_failure_as_a_transport_failure() {
    tokio::time::timeout(Duration::from_secs(5), async {
        // The response queue runs out at the gate's config.json fetch, which is a transport
        // failure rather than a verdict about the artifact, so the public error must stay
        // download_failed instead of claiming the model is not executable.
        let transport = Arc::new(MetadataCountingTransport::new([
            repository_metadata_response(),
            qwen_tree_response(),
        ]));
        let journey = DownloadJourney::new(transport.clone()).await;

        journey
            .start_download_and_await_failure()
            .await
            .expect_public_error(DownloadJobPublicErrorCode::DownloadFailed);

        assert_eq!(
            transport.payload_request_count(),
            0,
            "a transport failure must still transfer no payload bytes"
        );
    })
    .await
    .expect("gate retrieval failure journey should remain bounded");
}

#[tokio::test]
async fn should_transfer_a_valid_qwen_manifest_past_the_gate() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let transport = Arc::new(MetadataCountingTransport::new([
            repository_metadata_response(),
            qwen_tree_response(),
            servable_qwen_config_response(),
            json_response(serde_json::json!({
                "weight_map": {"model.layers.0.weight": QWEN_SHARD_FILE}
            })),
        ]));
        let journey = DownloadJourney::new(transport.clone()).await;

        journey.start_download_and_await_transfer_attempt().await;

        assert!(
            transport.payload_request_count() >= 1,
            "a valid manifest must pass the gate and reach payload transfer"
        );
    })
    .await
    .expect("valid manifest journey should remain bounded");
}

struct DownloadJourney {
    /// Keeps the temporary state directory alive for the whole journey; dropping it would
    /// delete the job store underneath the coordinator's background task.
    _test_directory: TempDir,
    transport: Arc<MetadataCountingTransport>,
    coordinator: LibraryDownloadCoordinator,
}

impl DownloadJourney {
    async fn new(transport: Arc<MetadataCountingTransport>) -> Self {
        let test_directory = TempDir::new().expect("temporary directory should be available");
        let journey_transport = Arc::clone(&transport);
        let catalog = DownloadCatalog::parse_json(&catalog_json())
            .expect("fictional catalog should be valid");
        let coordinator = LibraryDownloadCoordinator::new(
            Arc::new(catalog),
            test_directory.path().join("models"),
            Arc::new(FixedCapacityQuery),
            transport.clone() as Arc<dyn HubTransport>,
            transport as Arc<dyn HubPayloadTransport>,
            Arc::new(IgnoreDiscoveryRefresh),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        );
        coordinator
            .recover_startup_state()
            .await
            .expect("fresh coordinator state should recover cleanly");
        Self {
            _test_directory: test_directory,
            transport: journey_transport,
            coordinator,
        }
    }

    async fn start_download_and_await_failure(&self) -> FailedJobWaiter {
        self.coordinator
            .start(REPOSITORY_ID)
            .await
            .expect("a fresh coordinator should accept the download");
        let failed_job = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if let Some(download_job) = self
                    .coordinator
                    .current_job()
                    .await
                    .expect("durable job lookup should succeed")
                    && download_job.state() == DownloadJobState::Failed
                {
                    break download_job;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("the download should fail in preflight");
        FailedJobWaiter { failed_job }
    }

    async fn start_download_and_await_transfer_attempt(&self) {
        self.coordinator
            .start(REPOSITORY_ID)
            .await
            .expect("a fresh coordinator should accept the download");
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.transport.payload_request_count() == 0 {
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("a passing gate must reach payload transfer");
    }
}

struct FailedJobWaiter {
    failed_job: astronomical_supervisor::DownloadJob,
}

impl FailedJobWaiter {
    fn expect_public_error(&self, expected_error_code: DownloadJobPublicErrorCode) {
        assert_eq!(self.failed_job.state(), DownloadJobState::Failed);
        assert_eq!(
            self.failed_job.error_code(),
            Some(expected_error_code),
            "unexpected failure code: {:?}",
            self.failed_job.error_code()
        );
    }
}

/// Serves scripted responses in script order and records how many requests each transport
/// surface received, so journeys can prove how far the preflight progressed.
struct MetadataCountingTransport {
    responses: Mutex<VecDeque<HubHttpResponse>>,
    metadata_requests: std::sync::atomic::AtomicUsize,
    payload_requests: std::sync::atomic::AtomicUsize,
}

impl MetadataCountingTransport {
    fn new(responses: impl IntoIterator<Item = HubHttpResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            metadata_requests: std::sync::atomic::AtomicUsize::new(0),
            payload_requests: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn metadata_request_count(&self) -> usize {
        self.metadata_requests
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    fn payload_request_count(&self) -> usize {
        self.payload_requests
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl HubTransport for MetadataCountingTransport {
    fn execute(&self, _request: HubHttpRequest) -> HubTransportFuture<'_> {
        self.metadata_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            self.responses
                .lock()
                .map_err(|_| HubTransportError::new("scripted response lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected Hub request"))
        })
    }
}

impl HubPayloadTransport for MetadataCountingTransport {
    fn execute_payload(&self, _request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        self.payload_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async move {
            Err(HubTransportError::new(
                "the executable gate must reject before payload transfer",
            ))
        })
    }
}

struct FixedCapacityQuery;

impl DiskCapacityQuery for FixedCapacityQuery {
    fn available_space_bytes(&self, _existing_same_volume_path: &Path) -> io::Result<u64> {
        Ok(1_000_000_000)
    }
}

struct IgnoreDiscoveryRefresh;

impl DownloadPublicationRefresh for IgnoreDiscoveryRefresh {
    fn refresh(
        &self,
        _published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

fn repository_metadata_response() -> HubHttpResponse {
    json_response(serde_json::json!({
        "id": REPOSITORY_ID,
        "sha": REVISION,
        "private": false,
        "gated": false
    }))
}

fn tree_response(file_entries: serde_json::Value) -> HubHttpResponse {
    json_response(file_entries)
}

fn qwen_tree_response() -> HubHttpResponse {
    tree_response(serde_json::json!([
        {"type":"file","size":9,"path":"config.json","oid":GIT_BLOB_SHA1},
        {"type":"file","size":12,"path":"tokenizer.json","oid":GIT_BLOB_SHA1},
        {"type":"file","size":7,"path":QWEN_SHARD_FILE,"oid":GIT_BLOB_SHA1},
        {"type":"file","size":11,"path":"model.safetensors.index.json","oid":GIT_BLOB_SHA1}
    ]))
}

fn flux_pipeline_tree_response() -> HubHttpResponse {
    tree_response(serde_json::json!([
        {"type":"file","size":9,"path":"config.json","oid":GIT_BLOB_SHA1},
        {"type":"file","size":60,"path":"model_index.json","oid":GIT_BLOB_SHA1}
    ]))
}

fn flux_pipeline_index_response() -> HubHttpResponse {
    json_response(serde_json::json!({
        "_class_name": "Flux2KleinPipeline",
        "is_distilled": true,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"]
    }))
}

fn servable_qwen_config_response() -> HubHttpResponse {
    json_response(serde_json::json!({
        "model_type": "qwen3_5",
        "text_config": {"max_position_embeddings": 262144}
    }))
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
        "{{\"schema_version\":2,\"entries\":[{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"display_name\":\"Example model\",\"family\":\"qwen3_5\",\"approximate_size_bytes\":39,\"public\":true}}]}}"
    )
}
