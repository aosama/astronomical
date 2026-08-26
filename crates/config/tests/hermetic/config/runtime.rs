//! Acceptance coverage for the first-run v1 document and instance policy.

use super::*;

#[test]
fn should_create_minimal_v1_config_and_byte_identical_local_schema_on_first_run() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("first run should create v1 configuration");
    let state_directory = temporary_home_directory.path().join(".astronomical");
    let generated_config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(state_directory.join("config.json"))
            .expect("first-run config should be readable"),
    )
    .expect("first-run config should be valid JSON");
    let generated_schema = std::fs::read(state_directory.join("astronomical-config.schema.json"))
        .expect("local schema should be readable");
    let checked_in_schema = std::fs::read(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../site/schemas/config/v1/astronomical-config.schema.json"),
    )
    .expect("checked-in schema should be readable");
    let checked_in_schema_json: serde_json::Value =
        serde_json::from_slice(&checked_in_schema).expect("checked-in schema should be valid JSON");

    assert_eq!(
        generated_config["$schema"],
        "./astronomical-config.schema.json"
    );
    assert_eq!(generated_config["schema_version"], 1);
    assert_eq!(
        generated_config["runtime"]["model_directories"],
        serde_json::json!([])
    );
    assert_eq!(
        generated_config.as_object().map(serde_json::Map::len),
        Some(4)
    );
    assert_eq!(generated_schema, checked_in_schema);
    assert_eq!(
        checked_in_schema_json["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(
        checked_in_schema_json["$id"],
        "https://aosama.github.io/astronomical/schemas/config/v1/astronomical-config.schema.json"
    );
    assert_eq!(checked_in_schema_json["additionalProperties"], false);
    assert_eq!(
        checked_in_schema_json["$defs"]["runtime"]["properties"]["model_directories"]["items"]["pattern"],
        "^/"
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["generation_defaults"]["properties"]["maximum_output_tokens"]
            ["maximum"],
        65_535
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["generation_defaults"]["properties"]["temperature"]["multipleOf"],
        0.001
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["generation_defaults"]["properties"]["top_p"]["multipleOf"],
        0.001
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["runtime"]["properties"]["maximum_mlx_memory_gb"]["maximum"],
        18_446_744_073_u64
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["prompt_cache"]["properties"]["maximum_size_gb"]["maximum"],
        18_446_744_073_u64
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["runtime"]["properties"]["experimental_qwen_thinking_channel_seed_enabled"]
            ["x-astronomical-apply-mode"],
        "worker-replacement"
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["diagnostics"]["properties"]["performance_attribution_enabled"]
            ["x-astronomical-apply-mode"],
        "application-restart"
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["mtp"]["properties"]["enabled"]["default"],
        false
    );
    assert_eq!(
        checked_in_schema_json["$defs"]["mtp"]["x-astronomical-apply-mode"],
        "model-reload"
    );
    assert!(astronomical_config.model_directories().is_empty());
    assert_eq!(
        astronomical_config.maximum_mlx_memory_bytes().unwrap(),
        None
    );
    assert_eq!(astronomical_config.chunking().unwrap(), Default::default());
    assert_eq!(
        generated_config["chunking"]["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"],
        2_048
    );
    assert_eq!(
        generated_config["chunking"]["prefill_graph_submission_layer_interval"],
        0
    );
    assert_eq!(
        generated_config["chunking"]["experimental_ssd_paging_prefill_graph_submission_layer_interval"],
        1
    );
    assert!(astronomical_config.persistent_prompt_cache_enabled());
    assert!(!astronomical_config.performance_attribution_enabled());
    assert!(!astronomical_config.experimental_qwen_thinking_channel_seed_enabled());
}

#[test]
fn should_enable_the_experimental_qwen_thinking_channel_seed_only_when_explicitly_configured() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[],"experimental_qwen_thinking_channel_seed_enabled":true}}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the explicit experimental flag should load");

    assert!(astronomical_config.experimental_qwen_thinking_channel_seed_enabled());
}

#[test]
fn should_reject_unknown_v1_top_level_and_runtime_fields() {
    for config_document in [
        r#"{"schema_version":1,"runtime":{"model_directories":[]},"unknown":true}"#,
        r#"{"schema_version":1,"runtime":{"model_directories":[],"bind_address":"127.0.0.1:7000"}}"#,
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(temporary_home_directory.path(), config_document);

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::ParseConfigFile { .. })
        ));
    }
}

#[test]
fn should_reject_relative_v1_model_directory_paths() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":["relative/model"]}}"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::PathMustBeAbsolute { .. })
    ));
}

#[test]
fn should_keep_configuration_generation_stable_across_formatting_only_changes() {
    let compact_home_directory = tempfile::tempdir().expect("compact config home should exist");
    let formatted_home_directory = tempfile::tempdir().expect("formatted config home should exist");
    let compact_document = r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"prompt_cache":{"enabled":true,"maximum_size_gb":50}}"#;
    let formatted_document = r#"{
        "prompt_cache": { "maximum_size_gb": 50, "enabled": true },
        "runtime": { "model_directories": [] },
        "schema_version": 1,
        "$schema": "./astronomical-config.schema.json"
    }"#;
    write_config(compact_home_directory.path(), compact_document);
    write_config(formatted_home_directory.path(), formatted_document);

    let compact_config =
        AstronomicalConfig::load_from_home_directory(compact_home_directory.path())
            .expect("compact configuration should load");
    let formatted_config =
        AstronomicalConfig::load_from_home_directory(formatted_home_directory.path())
            .expect("formatted configuration should load");

    assert_eq!(compact_config.generation(), formatted_config.generation());
    assert_eq!(compact_config.generation().len(), 64);
    assert!(
        compact_config
            .generation()
            .bytes()
            .all(|generation_byte| generation_byte.is_ascii_digit()
                || (b'a'..=b'f').contains(&generation_byte))
    );
}

#[test]
fn should_change_configuration_generation_after_a_semantic_change() {
    let first_home_directory = tempfile::tempdir().expect("first config home should exist");
    let second_home_directory = tempfile::tempdir().expect("second config home should exist");
    write_config(
        first_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"prompt_cache":{"enabled":true}}"#,
    );
    write_config(
        second_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"prompt_cache":{"enabled":false}}"#,
    );

    let first_config = AstronomicalConfig::load_from_home_directory(first_home_directory.path())
        .expect("first configuration should load");
    let second_config = AstronomicalConfig::load_from_home_directory(second_home_directory.path())
        .expect("second configuration should load");

    assert_ne!(first_config.generation(), second_config.generation());
}
