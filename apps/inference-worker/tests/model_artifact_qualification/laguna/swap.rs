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

use super::artifact::{
    LAGUNA_XS_PUBLIC_MODEL_ID, compact_romeo_and_juliet_source, resolve_reference_model_directory,
};
use super::http::assert_laguna_is_advertised;
use crate::model_artifact_qualification::model_artifact_rest_qualification::{
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
    let qwen_model = crate::common::configured_discovered_models()
        .into_iter()
        .filter(|model| model.model_family == astronomical_config::ModelFamily::Qwen3_5)
        .min_by_key(|model| model.model_size_bytes)
        .expect("Development roots should contain one executable Qwen model");
    let model_directories = HashMap::from([
        (qwen_model.model_id.clone(), qwen_model.model_directory),
        (
            LAGUNA_XS_PUBLIC_MODEL_ID.to_owned(),
            resolve_reference_model_directory(),
        ),
    ]);
    let rest_server = launch_multi_model_server(model_directories).await;
    assert_laguna_is_advertised(rest_server.server_address, LAGUNA_XS_PUBLIC_MODEL_ID).await;
    for (model_id, phase) in [
        (qwen_model.model_id.as_str(), "qwen"),
        (LAGUNA_XS_PUBLIC_MODEL_ID, "laguna-xs"),
        (qwen_model.model_id.as_str(), "qwen-return"),
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
            "temperature": 0,
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
    let isolated_home = crate::common::isolated_development_home_from_user_config();
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
