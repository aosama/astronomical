#![forbid(unsafe_code)]

//! Stand-in `astronomicald` for cross-process tests: the same HTTP and IPC
//! surfaces as the real daemon, but its worker is a scripted fixture binary,
//! so journeys drive the whole process tree without model artifacts. Setting
//! `ASTRONOMICAL_TEST_DAEMON_DISCOVERED_IMAGE_MODEL_ID` additionally injects
//! one discovered image-generation model so `/v1/models` advertisement and
//! the image-revision plumbing are exercised too.

use std::{
    collections::HashMap,
    env,
    error::Error,
    path::{Path, PathBuf},
    sync::{Arc, RwLock, atomic::AtomicU64},
    time::Duration,
};

use astronomical_config::{
    AstronomicalInstancePaths, AstronomicalRuntimeInstance, DiscoveredModel,
    ImageGenerationCapabilities, LogLevel, LoggingConfig, ModelCapabilities, ModelFamily,
    ModelLicense, PromptCacheConfig,
};
use astronomical_daemon::{
    DaemonIpcGenerationContext, DaemonPerformanceAttributionLog, DownloadCatalog,
    GenerationPerformanceLog, ResolvedRuntimeConfig, WorkerHandle, build_application,
    build_application_with_discovered_models, start_daemon_ipc_service,
};
use tokio::net::TcpListener;

const TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH";
const DISCOVERED_IMAGE_MODEL_ID_ENVIRONMENT_VARIABLE: &str =
    "ASTRONOMICAL_TEST_DAEMON_DISCOVERED_IMAGE_MODEL_ID";
/// Decimal SI bound (50 GB) for the fixture prompt-cache config; the scripted
/// worker never stores blocks, so only the wiring must be coherent.
const FIXTURE_PROMPT_CACHE_MAXIMUM_SIZE_BYTES: u64 = 50_000_000_000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let state_directory = state_directory_argument()?;
    let instance_paths = AstronomicalInstancePaths::for_state_directory(
        state_directory.clone(),
        AstronomicalRuntimeInstance::Development,
    );
    let daemon_bind_address = instance_paths.default_bind_address();
    let worker_executable_path =
        PathBuf::from(env::var(TEST_WORKER_EXECUTABLE_PATH_ENVIRONMENT_VARIABLE)?);
    let discovered_image_model = discovered_image_model_from_environment();
    let listener = TcpListener::bind(daemon_bind_address).await?;
    let bound_daemon_address = listener.local_addr()?;
    let performance_log_directory = instance_paths.logging_directory();
    std::fs::create_dir_all(&performance_log_directory)?;
    let performance_log = GenerationPerformanceLog::open(&performance_log_directory)?;
    let worker_handle = WorkerHandle::launch(
        worker_executable_path.clone(),
        Duration::from_secs(60),
        performance_log,
        Arc::new(HashMap::new()),
    )
    .await?;

    let ipc_service = start_daemon_ipc_service(
        &instance_paths,
        DaemonIpcGenerationContext {
            executor: Arc::new(worker_handle.clone()),
            reloadable_config: discovered_image_model.as_ref().map(|image_model| {
                Arc::new(RwLock::new(fixture_resolved_runtime_config(
                    &state_directory,
                    &instance_paths,
                    bound_daemon_address.to_string(),
                    worker_executable_path.clone(),
                    std::slice::from_ref(image_model),
                )))
            }),
            next_chat_request_id: Arc::new(AtomicU64::new(1)),
            download_catalog: Arc::new(DownloadCatalog::empty()),
            library_download_coordinator: None,
            instance_paths: instance_paths.clone(),
        },
        &DaemonPerformanceAttributionLog::disabled(),
    )
    .await?;

    let application = match discovered_image_model {
        Some(image_model) => {
            build_application_with_discovered_models(worker_handle.clone(), vec![image_model])
        }
        None => build_application(worker_handle.clone()),
    };

    println!("astronomicald listening on http://{bound_daemon_address}");

    let serve_result = axum::serve(listener, application)
        .with_graceful_shutdown(shutdown_signal())
        .await;
    let ipc_shutdown_result = ipc_service.shutdown().await;
    let worker_shutdown_result = worker_handle.shutdown().await;
    serve_result?;
    ipc_shutdown_result?;
    worker_shutdown_result?;

    Ok(())
}

/// Mirrors the image-generation model the daemon REST tests register (256 to
/// 1024 pixels in 32-pixel steps, Qwen-Image-2.1 family) under the requested
/// model ID, so journeys exercise the same discovery and revision plumbing.
fn discovered_image_model_from_environment() -> Option<DiscoveredModel> {
    let image_model_id = env::var(DISCOVERED_IMAGE_MODEL_ID_ENVIRONMENT_VARIABLE).ok()?;
    Some(DiscoveredModel {
        model_id: image_model_id,
        provider_model_id: Some("mlx-community/Qwen-Image-2.1".to_owned()),
        model_family: ModelFamily::QwenImage21,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/qwen-image-21"),
        capabilities: ModelCapabilities::ImageGeneration(ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing: false,
            supports_multiple_reference_images: false,
            default_steps: 40,
            minimum_dimension_pixels: 256,
            maximum_dimension_pixels: 1_024,
            dimension_multiple_pixels: 32,
        }),
        license: Some(ModelLicense::QwenResearch),
        model_size_bytes: 1,
    })
}

/// Builds the live configuration the IPC service consults, mirroring the real
/// daemon's wiring: the discovered model list, the worker path, and the cache
/// and log directories under the test state root. The model-policy catalog
/// stays empty because the scripted worker admits every generation.
fn fixture_resolved_runtime_config(
    state_directory: &Path,
    instance_paths: &AstronomicalInstancePaths,
    bind_address: String,
    worker_executable_path: PathBuf,
    discovered_models: &[DiscoveredModel],
) -> ResolvedRuntimeConfig {
    ResolvedRuntimeConfig {
        configuration_generation: "fixture-static-configuration-generation".to_owned(),
        worker_executable_path,
        discovered_models: discovered_models.to_vec(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::new()),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: true,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        prompt_cache_config: PromptCacheConfig::new(
            state_directory.join("prompt-cache"),
            FIXTURE_PROMPT_CACHE_MAXIMUM_SIZE_BYTES,
        ),
        bind_address,
        logging_config: LoggingConfig::new(instance_paths.logging_directory(), LogLevel::Warn, 7),
    }
}

fn state_directory_argument() -> Result<PathBuf, Box<dyn Error>> {
    let process_arguments = env::args_os().collect::<Vec<_>>();
    let state_directory_position = process_arguments
        .iter()
        .position(|argument| argument == "--state-directory")
        .ok_or("the test daemon requires --state-directory")?;
    process_arguments
        .get(state_directory_position + 1)
        .map(|state_directory| PathBuf::from(state_directory.as_os_str()))
        .ok_or_else(|| "the test daemon requires a state-directory value".into())
}

async fn shutdown_signal() {
    let interrupt_signal = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    let terminate_signal = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal_stream) => {
                signal_stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    tokio::select! {
        () = interrupt_signal => {},
        () = terminate_signal => {},
    }
}
