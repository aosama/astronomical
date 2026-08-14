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
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, ShutdownController, WorkerHandle,
    build_application_with_full_control,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::oneshot,
    task::JoinHandle,
    time::{Duration, sleep},
};

const READY_ATTEMPT_LIMIT: u8 = 70;
/// Leaves five seconds inside the repository-wide 120-second command maximum for cleanup.
pub(crate) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

pub(crate) struct RealModelRestServer {
    /// Owns and ultimately reaps the production worker subprocess.
    worker_handle: WorkerHandle,
    pub(crate) server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
}

pub(crate) async fn launch_real_model_rest_server(
    model_id: &str,
    model_directory: PathBuf,
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
    let mut worker_startup_configuration = runtime_config_resolver
        .load()
        .expect("the real-model worker configuration should resolve")
        .worker_startup_configuration();
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
        crate::common::single_model_directories(model_id, &model_directory),
        20_480,
        worker_startup_configuration,
    )
    .await
    .expect("the production worker should launch for real-model qualification");
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

pub(crate) async fn get_json_endpoint(
    server_address: SocketAddr,
    endpoint_path: &str,
) -> serde_json::Value {
    let request_text = format!(
        "GET {endpoint_path} HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n"
    );
    let response_text = send_http_request(server_address, request_text).await;
    let (_, response_body) = response_text
        .split_once("\r\n\r\n")
        .expect("the status response should contain HTTP headers");
    serde_json::from_str(response_body).expect("the status response body should be valid JSON")
}

async fn wait_until_ready(server_address: SocketAddr) {
    // Poll the public readiness endpoint with visible progress. Model loading can
    // legitimately take tens of seconds, and repository commands must never look
    // stalled while expensive real artifacts are being prepared.
    for readiness_attempt in 1..=READY_ATTEMPT_LIMIT {
        let request_text =
            format!("GET /ready HTTP/1.1\r\nHost: {server_address}\r\nConnection: close\r\n\r\n");
        if send_http_request(server_address, request_text)
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

async fn send_http_request(server_address: SocketAddr, request_text: String) -> String {
    // A tiny raw HTTP client avoids introducing another test server/client layer.
    // `Connection: close` makes EOF an unambiguous response boundary.
    let mut tcp_stream = TcpStream::connect(server_address)
        .await
        .expect("the real-model REST client should connect");
    tcp_stream
        .write_all(request_text.as_bytes())
        .await
        .expect("the real-model REST request should write");
    let mut response_bytes = Vec::new();
    tcp_stream
        .read_to_end(&mut response_bytes)
        .await
        .expect("the real-model REST response should read");
    String::from_utf8(response_bytes).expect("the real-model REST response should contain UTF-8")
}
