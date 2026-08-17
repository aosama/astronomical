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
    let config_file = home_directory.join(".astronomical-dev").join("config.json");
    std::fs::create_dir_all(config_file.parent().expect("config file has a parent"))
        .expect("the config directory should be created");
    std::fs::write(&config_file, config_body).expect("the config file should be written");
}

pub(super) fn sample_resolved_config() -> ResolvedRuntimeConfig {
    use std::collections::HashMap;
    ResolvedRuntimeConfig {
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_directories: Arc::new(HashMap::new()),
        max_output_tokens: 20_480,
        maximum_mlx_memory_bytes: None,
        chunking: astronomical_config::ChunkingConfig::default(),
        persistent_prompt_cache_enabled: true,
        performance_attribution_enabled: false,
        mtp_enabled: false,
        mtp_draft_depth: None,
        speculative_prefill: astronomical_config::SpeculativePrefillConfig::disabled(),
        speculative_prefill_draft_model_directory: None,
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
