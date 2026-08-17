use super::*;

#[test]
fn should_load_supported_runtime_settings_from_user_config_file() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": ["/models/ornith", "/models/qwen"],
          "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2048
          },
          "mtp_enabled": true,
          "mtp_draft_depth": 2,
          "supervisor": { "bind_address": "127.0.0.1:7000" },
          "logging": { "level": "debug", "retained_files": 3 }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config.model_directories(),
        [
            PathBuf::from("/models/ornith"),
            PathBuf::from("/models/qwen")
        ]
    );
    assert_eq!(
        astronomical_config
            .chunking()
            .expect("fixed chunking should resolve")
            .fixed_prompt_processing_chunk_size_tokens(),
        2_048
    );
    assert!(astronomical_config.mtp_enabled());
    assert_eq!(astronomical_config.mtp_draft_depth(), Some(2));
    assert_eq!(
        astronomical_config
            .supervisor_bind_address()
            .expect("bind address should resolve")
            .to_string(),
        "127.0.0.1:7000"
    );
    assert_eq!(
        astronomical_config
            .logging()
            .expect("logging should resolve"),
        LoggingConfig::new(
            temporary_home_directory.path().join(".astronomical/logs"),
            LogLevel::Debug,
            3,
        )
    );
}

#[test]
fn should_enable_mtp_when_omitted_from_user_config_file() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"model_directories": []}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(astronomical_config.mtp_enabled());
    assert_eq!(astronomical_config.mtp_draft_depth(), None);
}

#[test]
fn should_reject_configured_mtp_draft_depth_outside_one_through_three() {
    for invalid_depth in [0, 4] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(r#"{{"model_directories":[],"mtp_draft_depth":{invalid_depth}}}"#),
        );

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::InvalidMtpDraftDepth)
        ));
    }
}

#[test]
fn should_create_a_first_run_config_template_with_an_empty_model_directory_list() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("missing config should create the first-run template");
    let generated_config_text = std::fs::read_to_string(
        temporary_home_directory
            .path()
            .join(".astronomical/config.json"),
    )
    .expect("first-run template should be written");
    let generated_config_json: serde_json::Value =
        serde_json::from_str(&generated_config_text).expect("template should be valid JSON");

    assert!(astronomical_config.model_directories().is_empty());
    assert_eq!(
        generated_config_json.get("model_directories"),
        Some(&serde_json::json!([]))
    );
    assert_eq!(
        generated_config_json
            .get("prompt_cache_max_size_gb")
            .and_then(serde_json::Value::as_u64),
        Some(50)
    );
    assert_eq!(
        generated_config_json
            .get("mtp_enabled")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        generated_config_json
            .get("persistent_prompt_cache_enabled")
            .and_then(serde_json::Value::as_bool),
        Some(true)
    );
    assert_eq!(
        generated_config_json["chunking"]["fixed_prompt_processing_chunk_size_tokens"],
        2_048
    );
    assert_eq!(
        generated_config_json.get("chunking").and_then(|chunking| {
            chunking.get("experimental_ssd_paging_prefill_graph_submission_layer_interval")
        }),
        Some(&serde_json::json!(0))
    );
    assert_eq!(
        generated_config_json.get("chunking").and_then(|chunking| {
            chunking.get("experimental_ssd_paging_generation_graph_submission_layer_interval")
        }),
        Some(&serde_json::json!(3))
    );
    assert!(
        generated_config_json
            .get("chunking")
            .and_then(|chunking| chunking.get("prefill_graph_submission_layer_interval"))
            .is_none()
    );
    assert!(
        generated_config_json
            .get("chunking")
            .and_then(|chunking| chunking.get("generation_graph_submission_layer_interval"))
            .is_none()
    );
    assert!(astronomical_config.mtp_enabled());
    assert_eq!(
        astronomical_config
            .prompt_cache()
            .expect("first-run template should enable the prompt cache")
            .global_prompt_cache_maximum_size_bytes(),
        50_000_000_000
    );
}

#[test]
fn should_reject_relative_model_directory_paths() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"model_directories": ["relative/model"]}"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::PathMustBeAbsolute { .. })
    ));
}
