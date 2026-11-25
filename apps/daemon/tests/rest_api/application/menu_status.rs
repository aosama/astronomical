//! Keeps the complete menu fixture coupled to the production status endpoint for every loaded
//! model family. Swift consumes the same fixture through its real client and telemetry store.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use astronomical_daemon::{
    ResolvedRuntimeConfig, build_application, build_development_application_with_reload,
};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use ipc_protocol::{
    ModelLoadInFlight, WorkerImageGenerationModelConfiguration,
    WorkerLoadedAutoregressiveModelRuntimeConfiguration, WorkerLoadedModelRuntimeConfiguration,
    WorkerRuntimeFeatureConfiguration,
};
use serde_json::Value;
use tower::ServiceExt;

use crate::support::ScriptedExecutor;

const AUTOREGRESSIVE_MODEL_ID: &str = "fictional/autoregressive-model";
const AUTOREGRESSIVE_CONFIGURATION_GENERATION: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const FLUX_MODEL_ID: &str = "fictional/image-model";
const FLUX_CONFIGURATION_GENERATION: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const AUTOREGRESSIVE_STATUS_FIXTURE: &str = include_str!(
    "../../../../astronomical-menu/Tests/AstronomicalMenuContractTests/Fixtures/full-autoregressive-status.json"
);
const FLUX_STATUS_FIXTURE: &str = include_str!(
    "../../../../astronomical-menu/Tests/AstronomicalMenuContractTests/Fixtures/full-flux-status.json"
);

/// The prompt chunk defaults of the Metal build, which the menu app's fixtures
/// record. Non-macOS builds default to smaller chunks (see
/// `config::DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS`).
const METAL_BUILD_CHUNK_DEFAULTS: [(&str, u32, u32); 2] = [
    (
        "fixed_prompt_processing_chunk_size_tokens",
        config::DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
        2_048,
    ),
    (
        "fixed_ssd_streaming_prompt_processing_chunk_size_tokens",
        config::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
        4_096,
    ),
];

/// The macOS defaults are the Metal build's, which the fixtures record; a
/// change to either fails here and in the contracts below.
#[cfg(target_os = "macos")]
#[test]
fn should_keep_the_metal_build_chunk_defaults_on_macos() {
    for (chunk_field_name, platform_default, metal_build_default) in METAL_BUILD_CHUNK_DEFAULTS {
        assert_eq!(platform_default, metal_build_default, "{chunk_field_name}");
    }
}

/// Maps this build's chunk defaults to the Metal build's, so the fixtures of
/// the macOS menu app check every other field on every platform. On macOS
/// the document is compared as served, so a changed macOS default still
/// fails the contract until the fixtures change with it.
fn with_metal_build_chunk_defaults(mut status_document: Value) -> Value {
    if cfg!(target_os = "macos") {
        return status_document;
    }
    for (chunk_field_name, platform_default, metal_build_default) in METAL_BUILD_CHUNK_DEFAULTS {
        replace_chunk_default(
            &mut status_document,
            chunk_field_name,
            platform_default,
            metal_build_default,
        );
    }
    status_document
}

/// Replaces `platform_default` with `metal_build_default` wherever the field
/// holds it, directly or as the `default` and `effective` of a setting.
fn replace_chunk_default(
    status_value: &mut Value,
    chunk_field_name: &str,
    platform_default: u32,
    metal_build_default: u32,
) {
    match status_value {
        Value::Object(status_fields) => {
            for (field_name, field_value) in status_fields.iter_mut() {
                if field_name == chunk_field_name {
                    replace_default_value(field_value, platform_default, metal_build_default);
                }
                replace_chunk_default(
                    field_value,
                    chunk_field_name,
                    platform_default,
                    metal_build_default,
                );
            }
        }
        Value::Array(status_items) => {
            for status_item in status_items {
                replace_chunk_default(
                    status_item,
                    chunk_field_name,
                    platform_default,
                    metal_build_default,
                );
            }
        }
        _ => {}
    }
}

fn replace_default_value(field_value: &mut Value, platform_default: u32, metal_build_default: u32) {
    if field_value == &Value::from(platform_default) {
        *field_value = Value::from(metal_build_default);
        return;
    }
    if let Value::Object(setting_fields) = field_value {
        for setting_part in ["default", "effective"] {
            if setting_fields.get(setting_part) == Some(&Value::from(platform_default)) {
                setting_fields.insert(setting_part.to_owned(), Value::from(metal_build_default));
            }
        }
    }
}

#[tokio::test]
async fn should_keep_the_complete_autoregressive_menu_fixture_aligned_with_production_status() {
    let actual_status = production_status_document(autoregressive_runtime_configuration()).await;
    let expected_status = fixture_document(AUTOREGRESSIVE_STATUS_FIXTURE);

    assert_eq!(
        normalized_build_identity(
            with_metal_build_chunk_defaults(actual_status),
            &expected_status
        ),
        expected_status
    );
}

#[tokio::test]
async fn should_keep_the_complete_flux_menu_fixture_aligned_with_production_status() {
    let actual_status = production_status_document(flux_runtime_configuration()).await;
    let expected_status = fixture_document(FLUX_STATUS_FIXTURE);

    assert_eq!(
        normalized_build_identity(
            with_metal_build_chunk_defaults(actual_status),
            &expected_status
        ),
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
        neural_engine_program_directory: "neural-engine-programs".into(),
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/fictional/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::new()),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        astro_ssd_prompt_cache_enabled: true,
        configured_astro_ssd_prompt_cache_enabled: None,
        metal_residency_refresh_enabled: false,
        idle_model_unload_seconds: 0,
        menu_bar_shows_progress_and_speed: false,
        menu_bar_shows_technical_details: false,
        skill_settings: Default::default(),
        sharing_settings: Default::default(),
        configured_astro_ssd_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        astro_ssd_prompt_cache_config: config::AstroSsdPromptCacheConfig::new(
            PathBuf::from("/fictional/astro-ssd-prompt-cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: config::LoggingConfig::new(
            PathBuf::from("/fictional/astronomical-logs"),
            config::LogLevel::Warn,
            7,
        ),
    }
}

/// A swap reports `ready` throughout, so this field is all the menu has to go on.
#[tokio::test]
async fn should_report_a_load_in_flight_on_the_status_endpoint() {
    let mut scripted_executor = ScriptedExecutor::ready(Vec::new());
    scripted_executor.health_snapshot.model_load_in_flight =
        Some(ModelLoadInFlight::for_model("fictional/next-model"));

    let status_document = request_status_document(build_application(scripted_executor)).await;

    assert_eq!(status_document["status"], "ready");
    assert_eq!(
        status_document["model_load_in_flight"],
        serde_json::json!({ "model_id": "fictional/next-model" })
    );
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
        astro_ssd_prompt_cache_enabled: true,
        astro_ssd_prompt_cache_maximum_size_bytes: 10_000_000_000,
        loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                model_id: AUTOREGRESSIVE_MODEL_ID.to_owned(),
                maximum_context_tokens: 65_536,
                maximum_output_tokens: 8_192,
                chunking: crate::support::worker_chunking_with_prompt_chunk_tokens(2_048, 2_048),
            },
        )),
    }
}

fn flux_runtime_configuration() -> WorkerRuntimeFeatureConfiguration {
    WorkerRuntimeFeatureConfiguration {
        configuration_generation: FLUX_CONFIGURATION_GENERATION.to_owned(),
        astro_ssd_prompt_cache_enabled: false,
        astro_ssd_prompt_cache_maximum_size_bytes: 0,
        loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::ImageGeneration(
            WorkerImageGenerationModelConfiguration {
                model_id: FLUX_MODEL_ID.to_owned(),
                model_family: "flux2_klein".to_owned(),
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
