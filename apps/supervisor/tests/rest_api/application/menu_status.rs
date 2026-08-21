//! Keeps the complete menu fixture coupled to the production status endpoint for every loaded
//! model family. Swift consumes the same fixture through its real client and telemetry store.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use astronomical_ipc_protocol::{
    WorkerChunkingConfiguration, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerLoadedAutoregressiveModelRuntimeConfiguration,
    WorkerLoadedModelRuntimeConfiguration, WorkerRuntimeFeatureConfiguration,
};
use astronomical_supervisor::{
    ResolvedRuntimeConfig, build_application, build_development_application_with_reload,
};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use serde_json::Value;
use tower::ServiceExt;

use crate::common::ScriptedExecutor;

const AUTOREGRESSIVE_MODEL_ID: &str = "fictional/autoregressive-model";
const AUTOREGRESSIVE_CONFIGURATION_GENERATION: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const FLUX_MODEL_ID: &str = "fictional/image-model";
const FLUX_CONFIGURATION_GENERATION: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const AUTOREGRESSIVE_STATUS_FIXTURE: &str = include_str!(
    "../../../../astronomical-menu/tests/AstronomicalMenuContractTests/Fixtures/full-autoregressive-status.json"
);
const FLUX_STATUS_FIXTURE: &str = include_str!(
    "../../../../astronomical-menu/tests/AstronomicalMenuContractTests/Fixtures/full-flux-status.json"
);

#[tokio::test]
async fn should_keep_the_complete_autoregressive_menu_fixture_aligned_with_production_status() {
    let actual_status = production_status_document(autoregressive_runtime_configuration()).await;
    let expected_status = fixture_document(AUTOREGRESSIVE_STATUS_FIXTURE);

    assert_eq!(
        normalized_build_identity(actual_status, &expected_status),
        expected_status
    );
}

#[tokio::test]
async fn should_keep_the_complete_flux_menu_fixture_aligned_with_production_status() {
    let actual_status = production_status_document(flux_runtime_configuration()).await;
    let expected_status = fixture_document(FLUX_STATUS_FIXTURE);

    assert_eq!(
        normalized_build_identity(actual_status, &expected_status),
        expected_status
    );
}

#[tokio::test]
async fn should_report_the_standard_development_state_directory_required_by_the_menu() {
    let development_home_directory =
        tempfile::tempdir().expect("the Development fixture home should be created");
    let application = build_development_application_with_reload(
        ScriptedExecutor::ready(Vec::new()),
        Arc::new(RwLock::new(development_resolved_config())),
        development_home_directory.path().to_path_buf(),
    );

    let status_document = request_status_document(application).await;
    let menu_fixture = fixture_document(AUTOREGRESSIVE_STATUS_FIXTURE);

    assert_eq!(
        status_document["application"]["state_directory"],
        menu_fixture["application"]["state_directory"]
    );
}

fn development_resolved_config() -> ResolvedRuntimeConfig {
    ResolvedRuntimeConfig {
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/fictional/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::new()),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: true,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            PathBuf::from("/fictional/prompt-cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            PathBuf::from("/fictional/astronomical-logs"),
            astronomical_config::LogLevel::Warn,
            7,
        ),
    }
}

async fn production_status_document(
    worker_runtime_configuration: WorkerRuntimeFeatureConfiguration,
) -> Value {
    let ready_model_id = worker_runtime_configuration
        .loaded_model
        .as_ref()
        .expect("the menu contract fixture requires a loaded model")
        .model_id()
        .to_owned();
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.ready_model_id = Some(ready_model_id);
    scripted_executor.health_snapshot = scripted_executor
        .health_snapshot
        .with_worker_runtime_feature_configuration(worker_runtime_configuration);
    request_status_document(build_application(scripted_executor)).await
}

async fn request_status_document(application: Router) -> Value {
    let response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("the status request should be valid"),
        )
        .await
        .expect("the application should return a status response");
    let status_body = to_bytes(response.into_body(), 16 * 1_024)
        .await
        .expect("the status body should be readable");
    serde_json::from_slice(&status_body).expect("the status body should contain JSON")
}

fn autoregressive_runtime_configuration() -> WorkerRuntimeFeatureConfiguration {
    WorkerRuntimeFeatureConfiguration {
        configuration_generation: AUTOREGRESSIVE_CONFIGURATION_GENERATION.to_owned(),
        persistent_prompt_cache_enabled: true,
        prompt_cache_maximum_size_bytes: 10_000_000_000,
        loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                model_id: AUTOREGRESSIVE_MODEL_ID.to_owned(),
                maximum_context_tokens: 65_536,
                maximum_output_tokens: 8_192,
                chunking: WorkerChunkingConfiguration {
                    fixed_prompt_processing_chunk_size_tokens: 2_048,
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                    full_attention_key_value_growth_tokens: 256,
                    speculative_prefill_draft_forward_tokens: 2_048,
                    prefill_graph_submission_layer_interval: 1,
                    experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                    prompt_cache_block_tokens: None,
                    prompt_cache_common_prefix_stride_blocks: 4,
                },
                mtp_draft_depth: None,
                mtp_head_model_id: None,
                speculative_prefill_enabled: false,
                speculative_prefill: None,
            },
        )),
    }
}

fn flux_runtime_configuration() -> WorkerRuntimeFeatureConfiguration {
    WorkerRuntimeFeatureConfiguration {
        configuration_generation: FLUX_CONFIGURATION_GENERATION.to_owned(),
        persistent_prompt_cache_enabled: false,
        prompt_cache_maximum_size_bytes: 0,
        loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Flux2Klein(
            WorkerFlux2KleinModelConfiguration {
                model_id: FLUX_MODEL_ID.to_owned(),
                model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                artifact_revision: "fictional-revision".to_owned(),
            },
        )),
    }
}

fn fixture_document(fixture: &str) -> Value {
    serde_json::from_str(fixture).expect("the shared menu fixture should contain JSON")
}

fn normalized_build_identity(mut actual_status: Value, expected_status: &Value) -> Value {
    let actual_application = actual_status["application"]
        .as_object()
        .expect("production status should contain an application object");
    let expected_application = expected_status["application"]
        .as_object()
        .expect("the menu fixture should contain an application object");
    assert_eq!(actual_application.len(), expected_application.len());
    for expected_field_name in expected_application.keys() {
        assert!(actual_application.contains_key(expected_field_name));
    }
    assert_eq!(
        actual_application["channel"],
        expected_application["channel"]
    );
    assert_eq!(
        actual_application["channel_display_name"],
        expected_application["channel_display_name"]
    );
    for environment_specific_field_name in [
        "version",
        "build_number",
        "commit",
        "is_dirty",
        "state_directory",
    ] {
        assert_eq!(
            std::mem::discriminant(&actual_application[environment_specific_field_name]),
            std::mem::discriminant(&expected_application[environment_specific_field_name])
        );
    }

    // Build metadata varies in CI. This in-memory router reports custom state, while the focused
    // standard-instance test above owns the exact home-relative state-directory label.
    for environment_specific_field_name in [
        "version",
        "build_number",
        "commit",
        "is_dirty",
        "state_directory",
    ] {
        actual_status["application"][environment_specific_field_name] =
            expected_status["application"][environment_specific_field_name].clone();
    }
    actual_status
}
