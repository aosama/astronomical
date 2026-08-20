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

pub(super) fn sample_resolved_config() -> ResolvedRuntimeConfig {
    use std::collections::HashMap;
    ResolvedRuntimeConfig {
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
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
