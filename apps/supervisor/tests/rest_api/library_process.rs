//! Actual-daemon phase acceptance for the Library foundation.

use std::{path::Path, process::Stdio, time::Duration};

use tokio::{process::Command, time::timeout};

use super::daemon_process::{
    get_endpoint, spawn_actual_instance_daemon, terminate_daemon, write_instance_config,
};

#[tokio::test]
async fn should_serve_the_library_and_discover_instance_models_without_configured_roots() {
    timeout(Duration::from_secs(15), async {
        eprintln!("[library-process] preparing isolated instance");
        let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
            .expect("Cargo should provide the astronomicald executable path");
        let instance_state_directory =
            tempfile::tempdir().expect("an isolated instance state should be created");
        write_instance_config(instance_state_directory.path());
        write_fictional_model(instance_state_directory.path());
        assert_fictional_model_is_structurally_discoverable(instance_state_directory.path());

        eprintln!("[library-process] starting astronomicald");
        let (mut daemon_process, daemon_address) = spawn_actual_instance_daemon(
            &daemon_executable_path,
            "development",
            instance_state_directory.path(),
        )
        .await;

        eprintln!("[library-process] checking catalog and Observatory");
        let catalog_response = get_endpoint(daemon_address, "/v1/library/catalog").await;
        assert!(catalog_response.starts_with("HTTP/1.1 200 OK"));
        let catalog_document: serde_json::Value =
            serde_json::from_str(http_response_body(&catalog_response))
                .expect("the process catalog response should contain JSON");
        assert_eq!(
            catalog_document,
            serde_json::json!({"schema_version": 1, "entries": []})
        );
        let library_response = get_endpoint(daemon_address, "/library").await;
        assert!(library_response.starts_with("HTTP/1.1 200 OK"));
        assert!(library_response.contains("data-observatory-view=\"library\""));

        eprintln!("[library-process] checking automatic model discovery");
        let models_response = get_endpoint(daemon_address, "/v1/models").await;
        assert!(models_response.starts_with("HTTP/1.1 200 OK"));
        let models_document: serde_json::Value =
            serde_json::from_str(http_response_body(&models_response))
                .expect("the process model response should contain JSON");
        assert!(
            models_document["data"]
                .as_array()
                .is_some_and(|models| models.iter().any(|model| model["id"] == "example-qwen"))
        );
        assert!(
            !instance_state_directory
                .path()
                .join("logs/supervisor-performance-attribution.jsonl")
                .exists()
        );

        eprintln!("[library-process] stopping astronomicald");
        terminate_daemon(&daemon_process);
        let daemon_status = timeout(Duration::from_secs(3), daemon_process.wait())
            .await
            .expect("the daemon should stop promptly")
            .expect("the daemon should be reaped");
        assert!(daemon_status.success());
    })
    .await
    .expect("the Library process journey must finish within fifteen seconds");
}

fn http_response_body(http_response: &str) -> &str {
    http_response
        .split_once("\r\n\r\n")
        .map(|(_, response_body)| response_body)
        .expect("the HTTP response should separate headers from its body")
}

#[tokio::test]
async fn should_record_the_startup_catalog_load_when_attribution_is_enabled() {
    timeout(Duration::from_secs(15), async {
        let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
            .expect("Cargo should provide the astronomicald executable path");
        let instance_state_directory =
            tempfile::tempdir().expect("an isolated instance state should be created");
        write_instance_config_with_performance_attribution(instance_state_directory.path());

        let (mut daemon_process, _) = spawn_actual_instance_daemon(
            &daemon_executable_path,
            "development",
            instance_state_directory.path(),
        )
        .await;

        let attribution_text = std::fs::read_to_string(
            instance_state_directory
                .path()
                .join("logs/supervisor-performance-attribution.jsonl"),
        )
        .expect("startup must flush the required catalog attribution record before binding");
        let attribution_record: serde_json::Value = serde_json::from_str(attribution_text.trim())
            .expect("the startup attribution record should contain JSON");
        assert_eq!(attribution_record["operation"], "library_catalog_load");
        assert_eq!(attribution_record["outcome"], "success");
        assert_eq!(attribution_record["catalog_entry_count"], 0);

        terminate_daemon(&daemon_process);
        assert!(
            timeout(Duration::from_secs(3), daemon_process.wait())
                .await
                .expect("the daemon should stop promptly")
                .expect("the daemon should be reaped")
                .success()
        );
    })
    .await
    .expect("enabled startup attribution must finish within fifteen seconds");
}

#[tokio::test]
async fn should_fail_before_binding_when_the_startup_attribution_file_cannot_open() {
    timeout(Duration::from_secs(10), async {
        let daemon_executable_path = std::env::var("CARGO_BIN_EXE_astronomicald")
            .expect("Cargo should provide the astronomicald executable path");
        let instance_state_directory =
            tempfile::tempdir().expect("an isolated instance state should be created");
        write_instance_config_with_performance_attribution(instance_state_directory.path());
        std::fs::create_dir_all(
            instance_state_directory
                .path()
                .join("logs/supervisor-performance-attribution.jsonl"),
        )
        .expect("a directory should occupy the required attribution file path");

        let daemon_output = Command::new(daemon_executable_path)
            .args(["--instance", "development", "--state-directory"])
            .arg(instance_state_directory.path())
            .env_remove("ASTRONOMICAL_TEST_WORKER_EXECUTABLE_PATH")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await
            .expect("the daemon process should run");

        assert!(!daemon_output.status.success());
        assert!(!String::from_utf8_lossy(&daemon_output.stdout).contains("listening on http://"));
        assert!(
            String::from_utf8_lossy(&daemon_output.stderr)
                .contains("failed to create the supervisor performance-attribution log")
        );
    })
    .await
    .expect("attribution initialization failure must stop startup promptly");
}

fn write_fictional_model(instance_state_directory: &Path) {
    const MODEL_SHARD_BYTES: &[u8] = b"fictional-shard";
    let model_directory = instance_state_directory
        .join("models")
        .join("astronomical-test")
        .join("example-qwen");
    std::fs::create_dir_all(&model_directory)
        .expect("the fictional model directory should be created");
    std::fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#,
    )
    .expect("the fictional model config should be written");
    std::fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("the fictional tokenizer should be written");
    std::fs::write(
        model_directory.join("model-00001.safetensors"),
        MODEL_SHARD_BYTES,
    )
    .expect("the fictional model shard should be written");
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        format!(
            r#"{{"metadata":{{"total_size":{}}},"weight_map":{{"model.embed_tokens.weight":"model-00001.safetensors"}}}}"#,
            MODEL_SHARD_BYTES.len()
        ),
    )
    .expect("the fictional model index should be written");
}

fn assert_fictional_model_is_structurally_discoverable(instance_state_directory: &Path) {
    let models_directory = instance_state_directory.join("models");
    let directory_scans = astronomical_config::discover_models(&[models_directory])
        .expect("the fictional automatic library should be discoverable");
    let discovered_model = directory_scans
        .first()
        .and_then(|directory_scan| directory_scan.discovered_models.first())
        .expect("the fictional model should pass structural discovery");
    let shard_size = std::fs::metadata(
        instance_state_directory
            .join("models/astronomical-test/example-qwen/model-00001.safetensors"),
    )
    .expect("the fictional shard metadata should be readable")
    .len();

    assert!(shard_size > 0);
    assert_eq!(discovered_model.model_size_bytes, shard_size);
}

fn write_instance_config_with_performance_attribution(state_directory: &Path) {
    std::fs::write(
        state_directory.join("config.json"),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"chunking":{"fixed_prompt_processing_chunk_size_tokens":2048},"diagnostics":{"performance_attribution_enabled":true}}"#,
    )
    .expect("instance config with performance attribution should be written");
}
