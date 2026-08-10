use super::*;

#[test]
fn should_load_supported_runtime_settings_from_user_config_file() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": ["/models/ornith", "/models/qwen"],
          "chunking": {
            "prefill_size_optimizer_enabled": false,
            "fixed_prefill_tokens": 2048
          },
          "mtp_enabled": true,
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
            .prefill_chunck_sizing_policy()
            .expect("fixed policy should resolve"),
        PrefillChunckSizingPolicy::Fixed {
            fixed_prefill_chunck_tokens: 2_048,
        }
    );
    assert!(astronomical_config.mtp_enabled());
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
        generated_config_json
            .get("chunking")
            .and_then(|chunking| chunking.get("optimizer_prefill_token_candidates")),
        Some(&serde_json::json!([1_024, 2_048, 4_096, 8_192]))
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
