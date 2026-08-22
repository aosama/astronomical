//! Acceptance coverage for strict v1 model policy and inheritance.

use super::*;

#[test]
fn should_resolve_full_v1_model_configuration_over_global_chunking() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema":"./astronomical-config.schema.json",
          "schema_version":1,
          "runtime":{"model_directories":[],"maximum_mlx_memory_gb":16},
          "prompt_cache":{"enabled":false,"maximum_size_gb":20},
          "chunking":{"fixed_prompt_processing_chunk_size_tokens":2048,"full_attention_key_value_growth_tokens":256},
          "models":{"organization/target":{"limits":{"maximum_context_tokens":32768},"generation_defaults":{"temperature":0.7,"top_p":0.9,"maximum_output_tokens":4096},"chunking":{"fixed_prompt_processing_chunk_size_tokens":4096},"acceleration":{"speculative_prefill":{"draft_model_id":"organization/draft","keep_percentage":30,"minimum_prompt_tokens":8192},"mtp":{"draft_depth":2}}}},
          "diagnostics":{"performance_attribution_enabled":true,"log_level":"debug","retained_log_files":3}
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("full v1 config should load");
    let model_config = astronomical_config
        .resolved_model_config("organization/target", 65_536)
        .expect("model policy should resolve");

    assert_eq!(model_config.maximum_context_tokens(), Some(32_768));
    assert_eq!(model_config.maximum_output_tokens(), 4_096);
    assert_eq!(model_config.temperature(), Some(0.7));
    assert_eq!(model_config.top_p(), Some(0.9));
    assert_eq!(
        model_config
            .chunking()
            .fixed_prompt_processing_chunk_size_tokens(),
        4_096
    );
    assert_eq!(
        model_config
            .chunking()
            .full_attention_key_value_growth_tokens(),
        256
    );
    assert_eq!(
        model_config
            .speculative_prefill()
            .expect("speculative prefill should be configured")
            .draft_model_id(),
        Some("organization/draft")
    );
    assert_eq!(model_config.mtp_draft_depth(), Some(2));
    assert_eq!(
        astronomical_config
            .maximum_mlx_memory_bytes()
            .expect("memory ceiling should resolve"),
        Some(16_000_000_000)
    );
    assert!(!astronomical_config.persistent_prompt_cache_enabled());
    assert_eq!(
        astronomical_config
            .prompt_cache()
            .expect("prompt cache should resolve")
            .global_prompt_cache_maximum_size_bytes(),
        20_000_000_000
    );
    assert!(astronomical_config.performance_attribution_enabled());
    assert_eq!(
        astronomical_config
            .logging()
            .expect("diagnostics should resolve")
            .level(),
        LogLevel::Debug
    );
}

#[test]
fn should_inherit_model_defaults_when_model_entry_or_properties_are_omitted() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"chunking":{"fixed_prompt_processing_chunk_size_tokens":4096,"prompt_cache_block_tokens":1024},"models":{"target":{"chunking":{"full_attention_key_value_growth_tokens":512,"prompt_cache_block_tokens":null}}}}"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("partial model config should load");

    let configured_model = astronomical_config
        .resolved_model_config("target", 65_536)
        .expect("partial model policy should resolve");
    let unconfigured_model = astronomical_config
        .resolved_model_config("another", 65_536)
        .expect("automatic model policy should resolve");

    assert_eq!(configured_model.maximum_output_tokens(), 20_480);
    assert_eq!(configured_model.temperature(), None);
    assert_eq!(configured_model.top_p(), None);
    assert_eq!(
        configured_model
            .chunking()
            .fixed_prompt_processing_chunk_size_tokens(),
        4_096
    );
    assert_eq!(
        configured_model
            .chunking()
            .full_attention_key_value_growth_tokens(),
        512
    );
    assert_eq!(
        configured_model.chunking().prompt_cache_block_tokens(),
        None
    );
    assert_eq!(
        unconfigured_model.chunking().prompt_cache_block_tokens(),
        Some(1_024)
    );
    assert!(configured_model.speculative_prefill().is_none());
    assert_eq!(configured_model.mtp_draft_depth(), None);
    assert_eq!(unconfigured_model.maximum_output_tokens(), 20_480);
}

#[test]
fn should_clamp_the_internal_output_default_for_a_small_context() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]}}"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("minimal config should load");

    let model_config = astronomical_config
        .resolved_model_config("small-model", 2_048)
        .expect("the internal default should not make a small model undiscoverable");

    assert_eq!(model_config.maximum_output_tokens(), 2_047);
    assert!(!model_config.has_explicit_maximum_output_tokens());
}

#[test]
fn should_identify_an_explicit_output_default_separately_from_internal_policy() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{"generation_defaults":{"maximum_output_tokens":128}}}}"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("configured output default should load");

    let model_config = astronomical_config
        .resolved_model_config("target", 2_048)
        .expect("configured output default should resolve");

    assert_eq!(model_config.maximum_output_tokens(), 128);
    assert!(model_config.has_explicit_maximum_output_tokens());
}

#[test]
fn should_apply_a_context_ceiling_not_exceeding_the_discovered_artifact() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{"limits":{"maximum_context_tokens":32768}}}}"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .resolved_model_config("target", 32_768)
            .expect("equal context ceilings should be valid")
            .maximum_context_tokens(),
        Some(32_768)
    );
    assert!(matches!(
        astronomical_config.resolved_model_config("target", 16_384),
        Err(AstronomicalConfigError::ConfiguredContextExceedsArtifact { .. })
    ));
}

#[test]
fn should_reject_an_explicit_output_default_that_does_not_fit_effective_context() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{"limits":{"maximum_context_tokens":2048},"generation_defaults":{"maximum_output_tokens":2048}}}}"#,
    );
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the structurally valid config should load before artifact resolution");

    assert!(matches!(
        astronomical_config.resolved_model_config("target", 65_536),
        Err(AstronomicalConfigError::ConfiguredOutputNotSmallerThanContext { .. })
    ));
}

#[test]
fn should_reject_invalid_model_ranges_and_unknown_nested_fields() {
    for invalid_model_config in [
        r#"{"generation_defaults":{"temperature":2.1}}"#,
        r#"{"generation_defaults":{"temperature":0.0001}}"#,
        r#"{"generation_defaults":{"top_p":1.1}}"#,
        r#"{"generation_defaults":{"top_p":0.0001}}"#,
        r#"{"generation_defaults":{"maximum_output_tokens":0}}"#,
        r#"{"generation_defaults":{"maximum_output_tokens":65536}}"#,
        r#"{"limits":{"maximum_context_tokens":0}}"#,
        r#"{"limits":{"maximum_context_tokens":1}}"#,
        r#"{"acceleration":{"mtp":{"draft_depth":4}}}"#,
        r#"{"acceleration":{"speculative_prefill":{"draft_model_id":"draft","keep_percentage":0}}}"#,
        r#"{"unknown":1}"#,
        r#"{"acceleration":{"unknown":1}}"#,
        r#"{"generation_defaults":{"unknown":1}}"#,
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(
                r#"{{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{{"model_directories":[]}},"models":{{"target":{invalid_model_config}}}}}"#
            ),
        );

        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect_err("invalid strict model policy must fail");
    }
}

#[test]
fn should_reject_the_retired_standalone_mtp_head_configuration() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{"$schema":"./astronomical-config.schema.json","schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{"acceleration":{"mtp":{"head_model_id":"organization/mtp"}}}}}"#,
    );

    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect_err("the strict schema must reject standalone MTP head selection");
}

#[test]
fn should_reject_control_characters_in_model_relationship_identities() {
    for configured_models in [
        serde_json::json!({"target\nmodel": {}}),
        serde_json::json!({"target": {"acceleration": {"speculative_prefill": {"draft_model_id": "draft\nmodel"}}}}),
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(
            temporary_home_directory.path(),
            &serde_json::json!({
                "$schema": "./astronomical-config.schema.json",
                "schema_version": 1,
                "runtime": {"model_directories": []},
                "models": configured_models,
            })
            .to_string(),
        );

        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect_err("model relationship identities must reject control characters");
    }
}

#[test]
fn should_reject_duplicate_keys_at_every_object_depth() {
    for duplicate_document in [
        r#"{"schema_version":1,"schema_version":1,"runtime":{"model_directories":[]}}"#,
        r#"{"schema_version":1,"runtime":{"model_directories":[],"model_directories":[]}}"#,
        r#"{"schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{},"target":{}}}"#,
        r#"{"schema_version":1,"runtime":{"model_directories":[]},"models":{"target":{"limits":{"maximum_context_tokens":1,"maximum_context_tokens":2}}}}"#,
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(temporary_home_directory.path(), duplicate_document);

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::DuplicateConfigKey { .. })
        ));
    }
}
