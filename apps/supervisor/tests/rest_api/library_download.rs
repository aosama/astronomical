//! Complete REST journey for daemon-owned download, verification, and publication.

use std::{
    collections::VecDeque,
    io,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use astronomical_supervisor::{
    DiskCapacityQuery, DownloadCatalog, DownloadPublicationRefresh, HubHttpRequest,
    HubHttpResponse, HubPayloadFuture, HubPayloadRequest, HubPayloadResponse, HubPayloadTransport,
    HubTransport, HubTransportError, HubTransportFuture, LibraryDownloadCoordinator,
    SupervisorPerformanceAttributionLog, build_application_with_library_download,
};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use bytes::Bytes;
use futures_util::stream;
use sha1::Sha1;
use sha2::Digest;
use tokio::time::timeout;
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
const MODEL_CONFIG: &[u8] =
    br#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#;
const CORRUPT_MODEL_CONFIG: &[u8] =
    br#"{"model_type":"qwen3_5_mof","text_config":{"max_position_embeddings":262144}}"#;
const TOKENIZER_CONFIG: &[u8] = br#"{"version":1,"model":{"type":"BPE"}}"#;
const MODEL_WEIGHTS: &[u8] = b"fictional-shard";
const MODEL_INDEX: &[u8] = br#"{"metadata":{"total_size":15},"weight_map":{"model.embed_tokens.weight":"model-00001.safetensors"}}"#;

#[tokio::test]
async fn should_download_publish_and_report_ready_through_the_library_rest_journey() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let download_catalog = Arc::new(
            DownloadCatalog::parse_json(&format!(
                "{{\"schema_version\":1,\"entries\":[{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"display_name\":\"Example Qwen\",\"family\":\"qwen3_5\",\"approximate_size_bytes\":{},\"public\":true}}]}}",
                repository_bytes()
            ))
            .expect("fictional catalog should parse"),
        );
        let hub = Arc::new(ScriptedHub::new());
        let coordinator = Arc::new(LibraryDownloadCoordinator::new(
            Arc::clone(&download_catalog),
            test_directory.path().join("models"),
            Arc::new(AvailableCapacity),
            hub.clone(),
            hub,
            Arc::new(ValidatingDiscoveryRefresh),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        ));
        let application = build_application_with_library_download(
            ScriptedExecutor::unavailable(),
            download_catalog,
            Arc::clone(&coordinator),
        );

        let start_response = application
            .clone()
            .oneshot(
                Request::post("/v1/library/download")
                    .header("content-type", "application/json")
                    .body(Body::from(format!("{{\"huggingface_id\":\"{REPOSITORY_ID}\"}}")))
                    .expect("start request should be valid"),
            )
            .await
            .expect("start response should be available");
        assert_eq!(start_response.status(), StatusCode::ACCEPTED);

        let mut ready_catalog = None;
        for _poll_attempt in 0..100 {
            let catalog_response = application
                .clone()
                .oneshot(
                    Request::get("/v1/library/catalog")
                        .body(Body::empty())
                        .expect("catalog request should be valid"),
                )
                .await
                .expect("catalog response should be available");
            let catalog_document = response_json(catalog_response).await;
            if catalog_document["entries"][0]["ready_on_this_mac"] == true {
                ready_catalog = Some(catalog_document);
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        assert!(ready_catalog.is_some(), "published model should become ready");
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}/config.json"))
                .is_file()
        );
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}/tokenizer.json"))
                .is_file()
        );
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}/model-00001.safetensors"))
                .is_file()
        );
        let download_response = application
            .oneshot(
                Request::get("/v1/library/download")
                    .body(Body::empty())
                    .expect("download request should be valid"),
            )
            .await
            .expect("download response should be available");
        assert_eq!(response_json(download_response).await["state"], "idle");
    })
    .await
    .expect("Library REST journey should remain bounded");
}

#[tokio::test]
async fn should_remain_publishing_until_discovery_validates_the_published_model() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let download_catalog = test_catalog();
        let hub = Arc::new(ScriptedHub::new());
        let coordinator = Arc::new(LibraryDownloadCoordinator::new(
            Arc::clone(&download_catalog),
            test_directory.path().join("models"),
            Arc::new(AvailableCapacity),
            hub.clone(),
            hub,
            Arc::new(FailingDiscoveryRefresh),
            SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
                .expect("disabled attribution should construct"),
        ));
        let application = build_application_with_library_download(
            ScriptedExecutor::unavailable(),
            download_catalog,
            coordinator,
        );

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "publishing").await;
        let catalog_response = application
            .clone()
            .oneshot(
                Request::get("/v1/library/catalog")
                    .body(Body::empty())
                    .expect("catalog request should be valid"),
            )
            .await
            .expect("catalog response should be available");
        let catalog_document = response_json(catalog_response).await;

        assert_eq!(catalog_document["entries"][0]["ready_on_this_mac"], false);
        assert_eq!(
            catalog_document["entries"][0]["download_state"],
            "publishing"
        );
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}"))
                .is_dir(),
            "the atomic rename may complete before discovery accepts the model"
        );
    })
    .await
    .expect("discovery failure readiness journey should remain bounded");
}

#[tokio::test]
async fn should_pause_survive_a_new_coordinator_and_resume_the_same_rest_job() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = build_test_application(
            test_directory.path(),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "downloading").await;
        let pause_response = post_without_body(&application, "/v1/library/download/pause").await;
        assert_eq!(pause_response.status(), StatusCode::OK);
        assert_eq!(response_json(pause_response).await["state"], "paused");

        let restarted_application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));
        let resume_response =
            post_without_body(&restarted_application, "/v1/library/download/resume").await;
        assert_eq!(resume_response.status(), StatusCode::ACCEPTED);
        wait_for_download_state(&restarted_application, "idle").await;
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}/config.json"))
                .is_file()
        );
    })
    .await
    .expect("pause and restart REST journey should remain bounded");
}

#[tokio::test]
async fn should_cancel_and_remove_the_staging_tree_through_rest() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let application = build_test_application(
            test_directory.path(),
            Arc::new(ScriptedHub::delayed(Duration::from_millis(250))),
        );

        assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
        wait_for_download_state(&application, "downloading").await;
        let cancel_response = post_without_body(&application, "/v1/library/download/cancel").await;
        assert_eq!(cancel_response.status(), StatusCode::OK);
        wait_for_download_state(&application, "idle").await;
        assert!(!test_directory.path().join("models/.incomplete").exists());
    })
    .await
    .expect("cancel REST journey should remain bounded");
}

#[tokio::test]
async fn should_report_gated_and_checksum_failures_without_local_paths() {
    timeout(Duration::from_secs(5), async {
        for (hub, expected_error_code) in [
            (Arc::new(ScriptedHub::gated()), "download_gated"),
            (
                Arc::new(ScriptedHub::checksum_mismatch()),
                "checksum_mismatch",
            ),
        ] {
            let test_directory =
                tempfile::tempdir().expect("temporary directory should be available");
            let application = build_test_application(test_directory.path(), hub);

            assert_eq!(start_download(&application).await, StatusCode::ACCEPTED);
            let failed_job = wait_for_download_state(&application, "failed").await;
            assert_eq!(failed_job["error_code"], expected_error_code);
            assert!(!failed_job["error_code"].to_string().contains('/'));
            assert!(
                !test_directory
                    .path()
                    .join(format!("models/{REPOSITORY_ID}"))
                    .exists()
            );
        }
    })
    .await
    .expect("failure REST journeys should remain bounded");
}

#[tokio::test]
async fn should_redownload_the_invalid_file_when_resuming_after_checksum_error() {
    timeout(Duration::from_secs(5), async {
        let test_directory = tempfile::tempdir().expect("temporary directory should be available");
        let failed_application = build_test_application(
            test_directory.path(),
            Arc::new(ScriptedHub::checksum_mismatch()),
        );
        assert_eq!(
            start_download(&failed_application).await,
            StatusCode::ACCEPTED
        );
        let failed_job = wait_for_download_state(&failed_application, "failed").await;
        assert_eq!(failed_job["error_code"], "checksum_mismatch");
        assert!(failed_job["bytes_completed"].as_u64() < failed_job["bytes_total"].as_u64());

        let retry_application =
            build_test_application(test_directory.path(), Arc::new(ScriptedHub::new()));
        let resume_response =
            post_without_body(&retry_application, "/v1/library/download/resume").await;
        assert_eq!(resume_response.status(), StatusCode::ACCEPTED);
        wait_for_download_state(&retry_application, "idle").await;
        assert!(
            test_directory
                .path()
                .join(format!("models/{REPOSITORY_ID}"))
                .is_dir()
        );
    })
    .await
    .expect("checksum retry REST journey should remain bounded");
}

struct AvailableCapacity;

impl DiskCapacityQuery for AvailableCapacity {
    fn available_space_bytes(&self, _existing_same_volume_path: &Path) -> io::Result<u64> {
        Ok(1_000_000_000)
    }
}

struct ValidatingDiscoveryRefresh;

impl DownloadPublicationRefresh for ValidatingDiscoveryRefresh {
    fn refresh(
        &self,
        published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let models_directory = published_directory
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| io::Error::other("published fixture should have a models ancestor"))?;
        let discovered_models =
            astronomical_config::discover_models(&[models_directory.to_path_buf()])?;
        if discovered_models.iter().any(|directory_scan| {
            directory_scan
                .discovered_models
                .iter()
                .any(|model| model.model_directory == published_directory)
        }) {
            return Ok(());
        }
        Err(io::Error::other("published fixture should be structurally discoverable").into())
    }
}

struct FailingDiscoveryRefresh;

impl DownloadPublicationRefresh for FailingDiscoveryRefresh {
    fn refresh(
        &self,
        _published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err(io::Error::other("intentional discovery refresh failure").into())
    }
}

struct ScriptedHub {
    metadata_responses: Mutex<VecDeque<HubHttpResponse>>,
    payload_delay: Duration,
    corrupt_model_config: bool,
}

impl ScriptedHub {
    fn new() -> Self {
        Self::with_payload_behavior(Duration::ZERO, false)
    }

    fn delayed(payload_delay: Duration) -> Self {
        Self::with_payload_behavior(payload_delay, false)
    }

    fn checksum_mismatch() -> Self {
        Self::with_payload_behavior(Duration::ZERO, true)
    }

    fn gated() -> Self {
        Self {
            metadata_responses: Mutex::new(
                [HubHttpResponse::try_new(403, [], [b"{}".to_vec()])
                    .expect("gated response should be valid")]
                .into(),
            ),
            payload_delay: Duration::ZERO,
            corrupt_model_config: false,
        }
    }

    fn with_payload_behavior(payload_delay: Duration, corrupt_model_config: bool) -> Self {
        let config_git_blob_sha1 = git_blob_sha1_hex(MODEL_CONFIG);
        Self {
            metadata_responses: Mutex::new(
                [
                    HubHttpResponse::try_new(
                        200,
                        [],
                        [serde_json::to_vec(&serde_json::json!({
                            "id": REPOSITORY_ID,
                            "sha": REVISION,
                            "private": false,
                            "gated": false
                        }))
                        .expect("metadata fixture should serialize")],
                    )
                    .expect("metadata response should be valid"),
                    HubHttpResponse::try_new(
                        200,
                        [],
                        [serde_json::to_vec(&serde_json::json!([
                            {
                                "type": "file",
                                "size": MODEL_CONFIG.len(),
                                "path": "config.json",
                                "oid": config_git_blob_sha1
                            },
                            {
                                "type": "file",
                                "size": TOKENIZER_CONFIG.len(),
                                "path": "tokenizer.json",
                                "oid": git_blob_sha1_hex(TOKENIZER_CONFIG)
                            },
                            {
                                "type": "file",
                                "size": MODEL_WEIGHTS.len(),
                                "path": "model-00001.safetensors",
                                "oid": git_blob_sha1_hex(MODEL_WEIGHTS)
                            },
                            {
                                "type": "file",
                                "size": MODEL_INDEX.len(),
                                "path": "model.safetensors.index.json",
                                "oid": git_blob_sha1_hex(MODEL_INDEX)
                            }
                        ]))
                        .expect("tree fixture should serialize")],
                    )
                    .expect("tree response should be valid"),
                ]
                .into(),
            ),
            payload_delay,
            corrupt_model_config,
        }
    }
}

impl HubTransport for ScriptedHub {
    fn execute(&self, _request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            self.metadata_responses
                .lock()
                .map_err(|_| HubTransportError::new("metadata lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected metadata request"))
        })
    }
}

impl HubPayloadTransport for ScriptedHub {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            tokio::time::sleep(self.payload_delay).await;
            let payload = payload_for_request(&request, self.corrupt_model_config)?;
            Ok(HubPayloadResponse::new(
                200,
                None,
                Some(payload.len() as u64),
                Box::pin(stream::iter([Ok(Bytes::from_static(payload))])),
            ))
        })
    }
}

fn build_test_application(test_directory: &Path, hub: Arc<ScriptedHub>) -> Router {
    let download_catalog = test_catalog();
    let coordinator = Arc::new(LibraryDownloadCoordinator::new(
        Arc::clone(&download_catalog),
        test_directory.join("models"),
        Arc::new(AvailableCapacity),
        hub.clone(),
        hub,
        Arc::new(ValidatingDiscoveryRefresh),
        SupervisorPerformanceAttributionLog::open(test_directory, false)
            .expect("disabled attribution should construct"),
    ));
    build_application_with_library_download(
        ScriptedExecutor::unavailable(),
        download_catalog,
        coordinator,
    )
}

fn test_catalog() -> Arc<DownloadCatalog> {
    Arc::new(
        DownloadCatalog::parse_json(&format!(
            "{{\"schema_version\":1,\"entries\":[{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"display_name\":\"Example Qwen\",\"family\":\"qwen3_5\",\"approximate_size_bytes\":{},\"public\":true}}]}}",
            repository_bytes()
        ))
        .expect("fictional catalog should parse"),
    )
}

fn payload_for_request(
    request: &HubPayloadRequest,
    corrupt_model_config: bool,
) -> Result<&'static [u8], HubTransportError> {
    if request.url().ends_with("/config.json") {
        return Ok(if corrupt_model_config {
            CORRUPT_MODEL_CONFIG
        } else {
            MODEL_CONFIG
        });
    }
    if request.url().ends_with("/tokenizer.json") {
        return Ok(TOKENIZER_CONFIG);
    }
    if request.url().ends_with("/model-00001.safetensors") {
        return Ok(MODEL_WEIGHTS);
    }
    if request.url().ends_with("/model.safetensors.index.json") {
        return Ok(MODEL_INDEX);
    }
    Err(HubTransportError::new("unexpected payload request"))
}

const fn repository_bytes() -> usize {
    MODEL_CONFIG.len() + TOKENIZER_CONFIG.len() + MODEL_WEIGHTS.len() + MODEL_INDEX.len()
}

async fn start_download(application: &Router) -> StatusCode {
    application
        .clone()
        .oneshot(
            Request::post("/v1/library/download")
                .header("content-type", "application/json")
                .body(Body::from(format!(
                    "{{\"huggingface_id\":\"{REPOSITORY_ID}\"}}"
                )))
                .expect("start request should be valid"),
        )
        .await
        .expect("start response should be available")
        .status()
}

async fn post_without_body(application: &Router, path: &str) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::post(path)
                .body(Body::empty())
                .expect("control request should be valid"),
        )
        .await
        .expect("control response should be available")
}

async fn wait_for_download_state(application: &Router, expected_state: &str) -> serde_json::Value {
    for _poll_attempt in 0..100 {
        let response = application
            .clone()
            .oneshot(
                Request::get("/v1/library/download")
                    .body(Body::empty())
                    .expect("download status request should be valid"),
            )
            .await
            .expect("download status response should be available");
        let download_document = response_json(response).await;
        if download_document["state"] == expected_state {
            return download_document;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("download did not reach expected state {expected_state}");
}

fn git_blob_sha1_hex(payload: &[u8]) -> String {
    let mut digest = Sha1::new();
    digest.update(format!("blob {}\0", payload.len()).as_bytes());
    digest.update(payload);
    let digest_bytes = digest.finalize();
    digest_bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

async fn response_json(response: axum::response::Response) -> serde_json::Value {
    let response_bytes = to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("response body should be readable");
    serde_json::from_slice(&response_bytes).expect("response should contain JSON")
}
