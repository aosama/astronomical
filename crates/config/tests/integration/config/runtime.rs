//! Acceptance coverage for the first-run v1 document and instance policy.

use super::*;

struct FirstRunDocuments {
    astronomical_config: AstronomicalConfig,
    generated_config: serde_json::Value,
    generated_schema: Vec<u8>,
    checked_in_schema: Vec<u8>,
    checked_in_schema_json: serde_json::Value,
}

/// Loads every first-run artifact the three specification tests below verify.
/// The temporary home directory is read fully before the loader returns, so the
/// config instance stays valid without keeping the directory alive: every
/// accessor the tests use reads the in-memory v1 document, not the disk.
fn load_first_run_documents() -> FirstRunDocuments {
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

    FirstRunDocuments {
        astronomical_config,
        generated_config,
        generated_schema,
        checked_in_schema,
        checked_in_schema_json,
    }
}

/// Every field the first-run document must carry, as a JSON pointer plus its
/// expected value. One row per contract fact keeps the assertions scannable,
/// and a drifting pointer is named in the failure message.
#[test]
fn should_create_minimal_v1_config_document_on_first_run() {
    let first_run_documents = load_first_run_documents();
    let generated_config = &first_run_documents.generated_config;

    let expected_generated_fields: [(&str, serde_json::Value); 6] = [
        (
            "/$schema",
            serde_json::json!("./astronomical-config.schema.json"),
        ),
        ("/schema_version", serde_json::json!(1)),
        ("/runtime/model_directories", serde_json::json!([])),
        (
            "/chunking/fixed_ssd_streaming_prompt_processing_chunk_size_tokens",
            serde_json::json!(2_048),
        ),
        (
            "/chunking/prefill_graph_submission_layer_interval",
            serde_json::json!(0),
        ),
        (
            "/chunking/experimental_ssd_paging_prefill_graph_submission_layer_interval",
            serde_json::json!(1),
        ),
    ];
    for (json_pointer, expected_value) in expected_generated_fields {
        assert_eq!(
            generated_config.pointer(json_pointer),
            Some(&expected_value),
            "generated first-run config at {json_pointer}"
        );
    }
    assert_eq!(
        generated_config.as_object().map(serde_json::Map::len),
        Some(4)
    );
}

/// The schema the first run writes to disk must be byte-for-byte the checked-in
/// copy, and its identity metadata must pin the v1 location.
#[test]
fn should_copy_the_checked_in_schema_byte_for_byte_and_pin_its_identity() {
    let first_run_documents = load_first_run_documents();
    assert_eq!(
        first_run_documents.generated_schema,
        first_run_documents.checked_in_schema
    );

    let checked_in_schema_json = &first_run_documents.checked_in_schema_json;
    let identity_fields: [(&str, serde_json::Value); 3] = [
        (
            "/$schema",
            serde_json::json!("https://json-schema.org/draft/2020-12/schema"),
        ),
        (
            "/$id",
            serde_json::json!(
                "https://aosama.github.io/astronomical/schemas/config/v1/astronomical-config.schema.json"
            ),
        ),
        ("/additionalProperties", serde_json::json!(false)),
    ];
    for (json_pointer, expected_value) in identity_fields {
        assert_eq!(
            checked_in_schema_json.pointer(json_pointer),
            Some(&expected_value),
            "checked-in schema at {json_pointer}"
        );
    }
    assert!(
        checked_in_schema_json
            .pointer("/$defs/runtime/properties/experimental_qwen_thinking_channel_seed_enabled")
            .is_none(),
        "thinking-channel seed must stay expunged from the runtime schema"
    );
}

/// The generation, memory, and attribution constraints the schema pins, plus
/// the typed defaults the in-memory first-run config must expose.
#[test]
fn should_pin_generation_memory_and_attribution_constraints_in_the_checked_in_schema() {
    let first_run_documents = load_first_run_documents();
    let checked_in_schema_json = &first_run_documents.checked_in_schema_json;

    let constraint_fields: [(&str, serde_json::Value); 8] = [
        (
            "/$defs/runtime/properties/model_directories/items/pattern",
            serde_json::json!("^/"),
        ),
        (
            "/$defs/generation_defaults/properties/maximum_output_tokens/maximum",
            serde_json::json!(65_535),
        ),
        (
            "/$defs/generation_defaults/properties/temperature/multipleOf",
            serde_json::json!(0.001),
        ),
        (
            "/$defs/generation_defaults/properties/top_p/multipleOf",
            serde_json::json!(0.001),
        ),
        (
            "/$defs/runtime/properties/maximum_mlx_memory_gb/maximum",
            serde_json::json!(18_446_744_073_u64),
        ),
        (
            "/$defs/prompt_cache/properties/maximum_size_gb/maximum",
            serde_json::json!(18_446_744_073_u64),
        ),
        (
            "/$defs/diagnostics/properties/performance_attribution_enabled/x-astronomical-apply-mode",
            serde_json::json!("application-restart"),
        ),
        (
            "/$defs/diagnostics/properties/completion_attribution_enabled/x-astronomical-apply-mode",
            serde_json::json!("application-restart"),
        ),
    ];
    for (json_pointer, expected_value) in constraint_fields {
        assert_eq!(
            checked_in_schema_json.pointer(json_pointer),
            Some(&expected_value),
            "checked-in schema at {json_pointer}"
        );
    }

    let astronomical_config = &first_run_documents.astronomical_config;
    assert!(astronomical_config.model_directories().is_empty());
    assert_eq!(
        astronomical_config.maximum_mlx_memory_bytes().unwrap(),
        None
    );
    assert_eq!(astronomical_config.chunking().unwrap(), Default::default());
    assert!(astronomical_config.persistent_prompt_cache_enabled());
    assert!(!astronomical_config.performance_attribution_enabled());
    assert!(!astronomical_config.completion_attribution_enabled());
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
fn should_reject_relative_v1_model_download_directory_paths() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[],"model_download_directory":"relative/downloads"}}"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::PathMustBeAbsolute { .. })
    ));
}

#[test]
fn should_expose_configured_v1_model_download_directory_override() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[],"model_download_directory":"/opt/shared-models"}}"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("configuration with a download override should load");

    assert_eq!(
        astronomical_config.model_download_directory(),
        Some(&PathBuf::from("/opt/shared-models"))
    );
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
