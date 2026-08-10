use astronomical_config::{AstronomicalConfig, AstronomicalConfigError};

use super::write_config;

#[test]
fn should_reject_retired_expert_paging_enabled_field() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": { "prefill_size_optimizer_enabled": true },
          "expert_paging_enabled": true
        }"#,
    );

    let configuration_error =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect_err("the retired expert paging field must be rejected");

    let AstronomicalConfigError::ParseConfigFile { source, .. } = configuration_error else {
        panic!("the strict config schema should reject the retired field");
    };
    assert!(
        source
            .to_string()
            .contains("unknown field `expert_paging_enabled`"),
        "the parse source should identify the retired field, got {source}"
    );
}

#[test]
fn should_reject_retired_expert_weight_memory_cache_maximum_size_field() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": { "prefill_size_optimizer_enabled": true },
          "expert_weight_memory_cache_maximum_size_gb": 4
        }"#,
    );

    let configuration_error =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect_err("the retired expert retention cap field must be rejected");

    let AstronomicalConfigError::ParseConfigFile { source, .. } = configuration_error else {
        panic!("the strict config schema should reject the retired field");
    };
    assert!(
        source
            .to_string()
            .contains("unknown field `expert_weight_memory_cache_maximum_size_gb`"),
        "the parse source should identify the retired field, got {source}"
    );
}

#[test]
fn should_reject_every_retired_consumer_configuration_field() {
    for (retired_field_name, retired_field_configuration) in [
        ("model_directory", r#"{"model_directory":"/models"}"#),
        (
            "named model_directories",
            r#"{"model_directories":{"local":"/models"}}"#,
        ),
        (
            "adaptive_ram_growth_guard_enabled",
            r#"{"adaptive_ram_growth_guard_enabled":true}"#,
        ),
        (
            "full_attention_kv_state_growth_tokens",
            r#"{"full_attention_kv_state_growth_tokens":256}"#,
        ),
        (
            "spectre_attention_enabled",
            r#"{"spectre_attention_enabled":true}"#,
        ),
        (
            "nested prompt_cache",
            r#"{"prompt_cache":{"enabled":true,"max_size_gb":50}}"#,
        ),
        (
            "supervisor worker_executable_path",
            r#"{"supervisor":{"worker_executable_path":"/bin/worker"}}"#,
        ),
    ] {
        let temporary_home_directory =
            tempfile::tempdir().expect("temporary home should be created");
        write_config(temporary_home_directory.path(), retired_field_configuration);

        let configuration_error =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
                .expect_err("retired field {retired_field_name} must be rejected");

        assert!(
            matches!(
                configuration_error,
                AstronomicalConfigError::ParseConfigFile { .. }
            ),
            "retired field {retired_field_name} should fail strict parsing, got {configuration_error:?}"
        );
    }
}
