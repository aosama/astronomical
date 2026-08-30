//! Deep Laguna load failure preserves the previously healthy public model.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use astronomical_supervisor::{
    GenerationPerformanceLog, ResolvedRuntimeConfigResolver, WorkerHandle,
    build_development_application_with_reload,
};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::time::{sleep, timeout};

use super::validate::{
    compact_romeo_and_juliet_source, laguna_xs_public_model_id, resolve_reference_model_directory,
};
use crate::serving_acceptance::chat::openai_rest::{
    assert_successful_streaming_chat_response, get_endpoint, post_chat_completion,
};

const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const MALFORMED_MODEL_ID: &str = "Laguna-Deep-Malformed";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "proves an exact public Laguna load failure preserves a healthy loaded model"]
async fn should_return_model_load_failed_and_reuse_the_healthy_laguna_model() {
    timeout(JOURNEY_TIMEOUT, run_malformed_swap_journey())
        .await
        .expect("the malformed Laguna journey must finish within 115 seconds");
}

async fn run_malformed_swap_journey() {
    let healthy_directory = resolve_reference_model_directory();
    let malformed_home = tempfile::tempdir().expect("a malformed fixture home should be created");
    let malformed_directory = malformed_home.path().join(MALFORMED_MODEL_ID);
    create_shallow_valid_deep_invalid_artifact(&healthy_directory, &malformed_directory);
    let isolated_home = tempfile::tempdir().expect("an isolated Development home should exist");
    write_two_model_config(
        isolated_home.path(),
        &healthy_directory,
        &malformed_directory,
    );
    let worker_executable_path = PathBuf::from(
        std::env::var("CARGO_BIN_EXE_astronomical-inference-worker")
            .expect("Cargo should provide the worker executable"),
    );
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        isolated_home.path().to_path_buf(),
        worker_executable_path.clone(),
    );
    let resolved_config = resolver
        .load()
        .expect("shallow discovery should advertise both Laguna artifacts");
    assert!(
        resolved_config
            .discovered_models
            .iter()
            .any(|model| model.model_id == MALFORMED_MODEL_ID)
    );
    let model_policy_catalog = Arc::clone(&resolved_config.model_policy_catalog);
    let log_directory = isolated_home.path().join("logs");
    std::fs::create_dir_all(&log_directory).expect("the log directory should be created");
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        &worker_executable_path,
        Duration::from_secs(60),
        GenerationPerformanceLog::open(&log_directory).expect("the performance log should open"),
        model_policy_catalog,
        resolved_config.worker_startup_configuration(),
    )
    .await
    .expect("the production worker should launch");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("the listener should bind");
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
            eprintln!("[laguna-malformed] ready attempt={readiness_attempt}");
            break;
        }
        sleep(Duration::from_secs(1)).await;
    }

    let healthy_request = request_body(laguna_xs_public_model_id());
    let healthy_response = post_chat_completion(server_address, healthy_request.clone()).await;
    assert_successful_streaming_chat_response(&healthy_response);
    let malformed_response =
        post_chat_completion(server_address, request_body(MALFORMED_MODEL_ID)).await;
    assert!(
        malformed_response.starts_with("HTTP/1.1 503 Service Unavailable"),
        "{malformed_response}"
    );
    assert!(
        malformed_response.contains(r#""code":"model_load_failed""#),
        "{malformed_response}"
    );
    let reused_response = post_chat_completion(server_address, healthy_request).await;
    assert_successful_streaming_chat_response(&reused_response);

    let _shutdown_sent = shutdown_sender.send(());
    server_task
        .await
        .expect("the server task should not panic")
        .expect("the server should stop");
    worker_handle
        .shutdown()
        .await
        .expect("the worker should terminate");
}

fn create_shallow_valid_deep_invalid_artifact(
    healthy_directory: &Path,
    malformed_directory: &Path,
) {
    std::fs::create_dir_all(malformed_directory.join(".cache/huggingface/download"))
        .expect("malformed fixture directories should be created");
    for directory_entry in
        std::fs::read_dir(healthy_directory).expect("the healthy artifact should be readable")
    {
        let entry_path = directory_entry
            .expect("artifact entries should be readable")
            .path();
        if !entry_path.is_file() {
            continue;
        }
        let file_name = entry_path
            .file_name()
            .expect("artifact files should have names");
        let destination_path = malformed_directory.join(file_name);
        if entry_path
            .extension()
            .and_then(|extension| extension.to_str())
            == Some("safetensors")
        {
            std::fs::hard_link(&entry_path, destination_path)
                .expect("malformed fixture shards should reuse healthy payloads");
        } else {
            std::fs::copy(&entry_path, destination_path)
                .expect("malformed fixture sidecars should be copied");
        }
    }
    let config_path = malformed_directory.join("config.json");
    let mut config_document: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path).expect("the copied config should be readable"),
    )
    .expect("the copied config should contain JSON");
    if let Some(language_config) = config_document.get_mut("text_config") {
        language_config["hidden_size"] = json!(0);
    } else {
        config_document["hidden_size"] = json!(0);
    }
    std::fs::write(
        config_path,
        serde_json::to_vec(&config_document).expect("the malformed config should serialize"),
    )
    .expect("the malformed config should be written");
    std::fs::copy(
        healthy_directory.join(".cache/huggingface/download/config.json.metadata"),
        malformed_directory.join(".cache/huggingface/download/config.json.metadata"),
    )
    .expect("the malformed fixture should retain the reference artifact revision");
}

fn write_two_model_config(home_directory: &Path, healthy: &Path, malformed: &Path) {
    let state_directory = home_directory.join(".astronomical-dev");
    std::fs::create_dir_all(&state_directory).expect("the isolated state should be created");
    std::fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec(&json!({
            "model_directories": [healthy, malformed],
            "max_output_tokens": 8,
            "mtp_enabled": false
        }))
        .expect("the isolated configuration should serialize"),
    )
    .expect("the isolated configuration should write");
}

fn request_body(model_id: &str) -> String {
    json!({
        "model": model_id,
        "messages": [{"role": "user", "content": format!(
            "Use the supplied Romeo and Juliet source. Name the households.\n\n{}",
            compact_romeo_and_juliet_source()
        )}],
        "stream": true,
        "temperature": 1,
        "max_tokens": 2
    })
    .to_string()
}
