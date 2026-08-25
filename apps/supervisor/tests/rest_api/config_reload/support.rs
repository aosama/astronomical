use super::*;

pub(super) async fn post_config_reload(application: &axum::Router) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/config/reload")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::empty())
                .expect("the reload request should be valid"),
        )
        .await
        .expect("the application should return a reload response")
}

pub(super) fn write_config_file(home_directory: &std::path::Path, config_body: &str) {
    let configured_fields: serde_json::Value =
        serde_json::from_str(config_body).expect("the partial v1 fixture should be valid JSON");
    let mut config_document = serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [] }
    });
    let configured_fields = configured_fields
        .as_object()
        .expect("the partial v1 fixture should be an object");
    let config_document_fields = config_document
        .as_object_mut()
        .expect("the base v1 fixture should be an object");
    for (field_name, field_value) in configured_fields {
        config_document_fields.insert(field_name.clone(), field_value.clone());
    }
    write_raw_config_file(
        home_directory,
        &serde_json::to_string_pretty(&config_document).expect("the v1 fixture should serialize"),
    );
}

pub(super) fn write_raw_config_file(home_directory: &std::path::Path, config_body: &str) {
    let config_file = home_directory.join(".astronomical-dev").join("config.json");
    std::fs::create_dir_all(config_file.parent().expect("config file has a parent"))
        .expect("the config directory should be created");
    std::fs::write(&config_file, config_body).expect("the config file should be written");
}

pub(super) fn write_minimal_qwen_model(model_directory: &std::path::Path) {
    const MODEL_SHARD_BYTES: &[u8] = b"fictional-shard";
    std::fs::create_dir_all(model_directory).expect("the model directory should be created");
    std::fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#,
    )
    .expect("the model config should be written");
    std::fs::write(
        model_directory.join("model-00001.safetensors"),
        MODEL_SHARD_BYTES,
    )
    .expect("the model shard should be written");
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        format!(
            r#"{{"metadata":{{"total_size":{}}},"weight_map":{{"model.embed_tokens.weight":"model-00001.safetensors"}}}}"#,
            MODEL_SHARD_BYTES.len()
        ),
    )
    .expect("the model index should be written");
    std::fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("the tokenizer should be written");
}

pub(super) fn sample_resolved_config() -> ResolvedRuntimeConfig {
    use std::collections::HashMap;
    ResolvedRuntimeConfig {
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::new()),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: true,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        experimental_qwen_thinking_channel_seed_enabled: false,
        prompt_cache_config: astronomical_config::PromptCacheConfig::new(
            PathBuf::from("/tmp/prompt-cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: astronomical_config::LoggingConfig::new(
            PathBuf::from("/tmp/astronomical-logs"),
            astronomical_config::LogLevel::Warn,
            7,
        ),
    }
}

pub(super) async fn post_shutdown(application: &axum::Router) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/control/shutdown")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::empty())
                .expect("the shutdown request should be valid"),
        )
        .await
        .expect("the application should return a shutdown response")
}
