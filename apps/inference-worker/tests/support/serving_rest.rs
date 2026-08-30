//! Shared production-boundary harness for real-model REST acceptance journeys.
//!
//! These tests do not call the model object directly. They launch the real worker
//! subprocess, connect it to the real supervisor/router, send public HTTP requests,
//! and observe public status. This is the widest local boundary that proves a user
//! can actually serve a configured model after memory-policy changes.
//!
//! Every journey supplies an isolated application home so configuration, logs,
//! prompt-cache files, and attribution cannot leak between runs. Cargo supplies
//! the just-built worker executable path; no developer-machine path is embedded.

use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, ShutdownController, WorkerHandle,
    build_application_with_discovered_models, build_application_with_full_control,
    build_development_application_with_reload,
};
use tokio::{
    net::TcpListener,
    sync::oneshot,
    task::JoinHandle,
    time::{Duration, sleep},
};

const READY_ATTEMPT_LIMIT: u8 = 70;
/// Leaves five seconds inside the repository-wide 120-second command maximum for cleanup.
#[allow(dead_code)]
pub(crate) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

#[allow(dead_code)]
pub(crate) struct RealModelRestServer {
    /// Owns and ultimately reaps the production worker subprocess.
    worker_handle: WorkerHandle,
    pub(crate) server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
}

#[allow(dead_code)]
pub(crate) async fn launch_real_model_rest_server(
    model_id: &str,
    model_directory: PathBuf,
    isolated_worker_home_directory: &Path,
    maximum_mlx_memory_bytes: u64,
) -> RealModelRestServer {
    launch_real_model_rest_server_for_models(
        &[(model_id.to_owned(), model_directory)],
        isolated_worker_home_directory,
        maximum_mlx_memory_bytes,
    )
    .await
}

#[allow(dead_code)]
pub(crate) async fn launch_real_model_rest_server_for_models(
    model_artifacts: &[(String, PathBuf)],
    isolated_worker_home_directory: &Path,
    maximum_mlx_memory_bytes: u64,
) -> RealModelRestServer {
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    // Resolve the same standard configuration used by production, but rooted in
    // the journey's temporary home. The explicit byte ceiling below is repeated
    // in startup IPC so the test cannot accidentally exercise a machine default.
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home_directory.to_path_buf(),
        production_worker_executable_path.clone(),
    );
    let resolved_runtime_config = runtime_config_resolver
        .load()
        .expect("the real-model worker configuration should resolve");
    let model_policy_catalog = Arc::new(
        model_artifacts
            .iter()
            .map(|(model_id, model_directory)| {
                let mut model_policy = resolved_runtime_config
                    .model_policy_catalog
                    .get(model_id)
                    .unwrap_or_else(|| {
                        panic!("the resolved policy catalog should include {model_id}")
                    })
                    .clone();
                model_policy.model_directory = model_directory.clone();
                (model_id.clone(), model_policy)
            })
            .collect::<HashMap<_, _>>(),
    );
    let mut worker_startup_configuration = resolved_runtime_config.worker_startup_configuration();
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(maximum_mlx_memory_bytes);
    // Keep the supervisor record beside the isolated worker logs so acceptance
    // journeys can reconcile public request timing with worker attribution after
    // an orderly stop. The entire directory remains owned by the test tempdir.
    let performance_log_directory = runtime_config_resolver.instance_paths().logging_directory();
    fs::create_dir_all(&performance_log_directory)
        .expect("the isolated real-model performance log directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &production_worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the real-model performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the production worker should launch for real-model acceptance");
    // Port zero asks the operating system for a free loopback port and avoids
    // collisions with a developer's ordinary Astronomical server.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the real-model REST listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the real-model REST listener should expose its address");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_worker_home_directory.to_path_buf(),
        production_worker_executable_path,
    );
    let resolved_runtime_config = runtime_config_resolver
        .load()
        .expect("the isolated real-model configuration should resolve");
    // Use the production daemon's full-control router so real-model journeys can
    // exercise public live-memory changes through HTTP instead of test-only
    // worker seams. The shutdown controller is owned only to satisfy that same
    // production boundary; this harness still performs explicit process cleanup.
    let application = build_application_with_full_control(
        worker_handle.clone(),
        Arc::new(RwLock::new(resolved_runtime_config)),
        runtime_config_resolver,
        ShutdownController::new(),
    );
    // The in-process HTTP server still talks to the out-of-process worker through
    // production IPC. Only socket/process ownership is kept in this test harness.
    let server = axum::serve(listener, application).with_graceful_shutdown(async {
        let _ = shutdown_receiver.await;
    });
    let server_task = tokio::spawn(async move { server.await });
    wait_until_ready(server_address).await;
    RealModelRestServer {
        worker_handle,
        server_address,
        shutdown_sender,
        server_task,
    }
}

#[allow(dead_code)]
pub(crate) async fn stop_real_model_rest_server(real_model_rest_server: RealModelRestServer) {
    // Stop accepting HTTP first, await graceful server completion, then ask the
    // worker handle to terminate and reap the subprocess. This order prevents a
    // late request from racing worker shutdown and leaving a child process alive.
    let _ = real_model_rest_server.shutdown_sender.send(());
    real_model_rest_server
        .server_task
        .await
        .expect("the real-model REST server task should not panic")
        .expect("the real-model REST server should stop cleanly");
    real_model_rest_server
        .worker_handle
        .shutdown()
        .await
        .expect("the real-model worker should terminate and be reaped");
}

#[allow(dead_code)]
pub(crate) async fn get_json_endpoint(
    server_address: SocketAddr,
    endpoint_path: &str,
) -> serde_json::Value {
    let request_text = format!(
        "GET {endpoint_path} HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n"
    );
    let response_text = super::http::send_http_request(server_address, request_text).await;
    let (_, response_body) = response_text
        .split_once("\r\n\r\n")
        .expect("the status response should contain HTTP headers");
    serde_json::from_str(response_body).expect("the status response body should be valid JSON")
}

#[allow(dead_code)]
pub(crate) async fn put_json_endpoint(
    server_address: SocketAddr,
    endpoint_path: &str,
    request_body: &serde_json::Value,
) -> serde_json::Value {
    let serialized_request_body =
        serde_json::to_string(request_body).expect("the real-model REST PUT body should serialize");
    let request_text = format!(
        "PUT {endpoint_path} HTTP/1.1\r\nHost: {server_address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{serialized_request_body}",
        serialized_request_body.len(),
    );
    let response_text = super::http::send_http_request(server_address, request_text).await;
    let (response_headers, response_body) = response_text
        .split_once("\r\n\r\n")
        .expect("the real-model REST PUT response should contain HTTP headers");
    let status_line = response_headers
        .lines()
        .next()
        .unwrap_or("missing status line");
    assert!(
        status_line.starts_with("HTTP/1.1 200 "),
        "the real-model REST PUT should succeed; status={status_line}; body={response_body}"
    );
    serde_json::from_str(response_body).unwrap_or_else(|response_parse_error| {
        panic!(
            "the successful real-model REST PUT body should be valid JSON: {response_parse_error}; status={status_line}; body={response_body}"
        )
    })
}

async fn wait_until_ready(server_address: SocketAddr) {
    // Poll the public readiness endpoint with visible progress. Model loading can
    // legitimately take tens of seconds, and repository commands must never look
    // stalled while expensive real artifacts are being prepared.
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let request_text =
            format!("GET /ready HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n");
        if super::http::send_http_request(server_address, request_text)
            .await
            .starts_with("HTTP/1.1 200 OK")
        {
            eprintln!("[real-model-rest] worker ready after {readiness_attempt} attempts");
            return;
        }
        let remaining_seconds = u16::from(READY_ATTEMPT_LIMIT - readiness_attempt);
        eprintln!(
            "[real-model-rest] loading attempt {readiness_attempt}/{READY_ATTEMPT_LIMIT}, ETA <= {remaining_seconds}s"
        );
        sleep(Duration::from_secs(1)).await;
    }
    panic!("the real-model worker did not become ready before the deadline");
}

pub(crate) struct ServingRestServer {
    worker_handle: WorkerHandle,
    pub(crate) server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
    isolated_development_home: Option<tempfile::TempDir>,
}

pub(crate) async fn launch_serving_rest_server_for_model(
    model_id: &str,
    model_directory: PathBuf,
    isolated_worker_home_directory: Option<&Path>,
    performance_log_directory: Option<&Path>,
) -> ServingRestServer {
    launch_serving_rest_server_for_model_with_memory_limit(
        model_id,
        model_directory,
        isolated_worker_home_directory,
        performance_log_directory,
        None,
    )
    .await
}

pub(crate) async fn launch_serving_rest_server_for_model_with_memory_limit(
    model_id: &str,
    model_directory: PathBuf,
    isolated_worker_home_directory: Option<&Path>,
    performance_log_directory: Option<&Path>,
    maximum_mlx_memory_bytes: Option<u64>,
) -> ServingRestServer {
    let production_worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the production inference-worker executable path"),
    );
    let owned_isolated_development_home = isolated_worker_home_directory
        .is_none()
        .then(super::isolated_development_home_from_user_config);
    let worker_configuration_home_directory = isolated_worker_home_directory
        .map(Path::to_path_buf)
        .or_else(|| {
            owned_isolated_development_home
                .as_ref()
                .map(|isolated_home| isolated_home.path().to_path_buf())
        })
        .expect("acceptance should own or receive an isolated Development home");
    let runtime_config_resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        worker_configuration_home_directory,
        production_worker_executable_path.clone(),
    );
    let worker_runtime_config = runtime_config_resolver
        .load()
        .expect("the serving REST worker configuration should resolve");
    let default_log_directory =
        tempfile::tempdir().expect("the serving REST log directory should be created");
    let performance_log_directory =
        performance_log_directory.unwrap_or_else(|| default_log_directory.path());
    let mut worker_startup_configuration = worker_runtime_config.worker_startup_configuration();
    if let Some(maximum_mlx_memory_bytes) = maximum_mlx_memory_bytes {
        worker_startup_configuration.configured_maximum_mlx_memory_bytes =
            Some(maximum_mlx_memory_bytes);
    }
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &production_worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(performance_log_directory)
            .expect("the serving REST performance log should open"),
        Arc::clone(&worker_runtime_config.model_policy_catalog),
        worker_startup_configuration,
    )
    .await
    .expect("the production worker should launch for serving REST");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the serving REST listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the serving REST listener should expose its address");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let application = match isolated_worker_home_directory {
        Some(isolated_worker_home_directory) => {
            let runtime_config_resolver =
                ResolvedRuntimeConfigResolver::for_development_home_directory(
                    isolated_worker_home_directory.to_path_buf(),
                    production_worker_executable_path,
                );
            let resolved_runtime_config = runtime_config_resolver
                .load()
                .expect("the isolated serving REST configuration should resolve");
            build_development_application_with_reload(
                worker_handle.clone(),
                Arc::new(RwLock::new(resolved_runtime_config)),
                isolated_worker_home_directory.to_path_buf(),
            )
        }
        None => build_application_with_discovered_models(
            worker_handle.clone(),
            vec![super::discovered_model_artifact(
                model_id,
                &model_directory,
                20_480,
            )],
        ),
    };
    let server = axum::serve(listener, application).with_graceful_shutdown(async {
        let _ = shutdown_receiver.await;
    });
    let server_task = tokio::spawn(async move { server.await });
    wait_until_ready(server_address).await;
    ServingRestServer {
        worker_handle,
        server_address,
        shutdown_sender,
        server_task,
        isolated_development_home: owned_isolated_development_home,
    }
}

pub(crate) async fn stop_serving_rest_server(serving_rest_server: ServingRestServer) {
    let ServingRestServer {
        worker_handle,
        shutdown_sender,
        server_task,
        isolated_development_home,
        ..
    } = serving_rest_server;
    let _ = shutdown_sender.send(());
    server_task
        .await
        .expect("the serving REST server task should not panic")
        .expect("the serving REST server should stop cleanly");
    worker_handle
        .shutdown()
        .await
        .expect("the serving worker should terminate and be reaped");
    drop(isolated_development_home);
}
