//! Same-worker Qwen to Laguna XS to Qwen public swap journey.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, WorkerHandle,
    build_development_application_with_reload,
};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

use super::http::assert_laguna_is_advertised;
use super::validate::{
    compact_romeo_and_juliet_source, laguna_xs_public_model_id, resolve_reference_model_directory,
};
use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, get_endpoint, post_chat_completion,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

struct MultiModelRestServer {
    worker_handle: WorkerHandle,
    server_address: SocketAddr,
    shutdown_sender: oneshot::Sender<()>,
    server_task: JoinHandle<Result<(), std::io::Error>>,
    _isolated_home: tempfile::TempDir,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "swaps Qwen, Laguna XS, then Qwen on one public worker"]
async fn should_swap_qwen_then_laguna_xs_then_qwen_on_one_worker() {
    timeout(JOURNEY_TIMEOUT, run_family_swap_journey())
        .await
        .expect("the complete family swap must finish within 115 seconds");
}

async fn run_family_swap_journey() {
    let small_dense_model_id = crate::support::small_dense_model_id();
    let qwen_model_directory =
        crate::support::configured_installed_model_directory_by_id(small_dense_model_id);
    let model_directories = HashMap::from([
        (small_dense_model_id.to_owned(), qwen_model_directory),
        (
            laguna_xs_public_model_id().to_owned(),
            resolve_reference_model_directory(),
        ),
    ]);
    let rest_server = launch_multi_model_server(model_directories).await;
    assert_laguna_is_advertised(rest_server.server_address, laguna_xs_public_model_id()).await;
    for (model_id, phase) in [
        (small_dense_model_id, "qwen"),
        (laguna_xs_public_model_id(), "laguna-xs"),
        (small_dense_model_id, "qwen-return"),
    ] {
        eprintln!("[laguna-swap] phase={phase} model={model_id}");
        let request_body = json!({
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": format!(
                    "Use the supplied Romeo and Juliet source. Name the households.\n\n{}",
                    compact_romeo_and_juliet_source()
                )
            }],
            "stream": true,
            "temperature": 1,
            "max_tokens": 2,
        })
        .to_string();
        let response = post_chat_completion(rest_server.server_address, request_body).await;
        assert_successful_streaming_chat_response(&response);
    }
    stop_multi_model_server(rest_server).await;
}

async fn launch_multi_model_server(
    model_directories: HashMap<String, PathBuf>,
) -> MultiModelRestServer {
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the worker executable"),
    );
    let isolated_home = crate::support::isolated_development_home_from_user_config();
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_home.path().to_path_buf(),
        worker_executable_path.clone(),
    );
    let resolved_config = resolver
        .load()
        .expect("the isolated family-swap configuration should resolve");
    let model_policy_catalog = Arc::new(
        model_directories
            .into_iter()
            .map(|(model_id, model_directory)| {
                let mut model_policy = resolved_config
                    .model_policy_catalog
                    .get(&model_id)
                    .unwrap_or_else(|| panic!("resolved policy should include {model_id}"))
                    .clone();
                model_policy.model_directory = model_directory;
                (model_id, model_policy)
            })
            .collect(),
    );
    let performance_log_directory = isolated_home.path().join("logs");
    std::fs::create_dir_all(&performance_log_directory)
        .expect("the isolated performance directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&performance_log_directory)
            .expect("the performance log should open"),
        model_policy_catalog,
        resolved_config.worker_startup_configuration(),
    )
    .await
    .expect("the family-swap worker should launch");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the family-swap listener should bind");
    let server_address = listener
        .local_addr()
        .expect("the listener should expose its address");
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let application = build_development_application_with_reload(
        worker_handle.clone(),
        Arc::new(RwLock::new(resolved_config)),
        isolated_home.path().to_path_buf(),
    );
    let server_task = tokio::spawn(async move {
        axum::serve(listener, application)
            .with_graceful_shutdown(async {
                let _ = shutdown_receiver.await;
            })
            .await
    });
    for readiness_attempt in 1..=70 {
        if get_endpoint(server_address, "/ready")
            .await
            .starts_with("HTTP/1.1 200 OK")
        {
            eprintln!("[laguna-swap] ready attempt={readiness_attempt}");
            break;
        }
        sleep(Duration::from_secs(1)).await;
    }
    MultiModelRestServer {
        worker_handle,
        server_address,
        shutdown_sender,
        server_task,
        _isolated_home: isolated_home,
    }
}

async fn stop_multi_model_server(rest_server: MultiModelRestServer) {
    let _shutdown_sent = rest_server.shutdown_sender.send(());
    rest_server
        .server_task
        .await
        .expect("the REST task should not panic")
        .expect("the REST server should stop cleanly");
    rest_server
        .worker_handle
        .shutdown()
        .await
        .expect("the family-swap worker should terminate");
}
