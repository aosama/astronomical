use super::*;

#[test]
fn should_disable_speculative_prefill_by_default() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(temporary_home_directory.path(), r#"{}"#);

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .speculative_prefill()
            .expect("speculative prefill config should resolve"),
        SpeculativePrefillConfig::disabled()
    );
}

#[test]
fn should_load_the_complete_speculative_prefill_policy() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_enabled": false,
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target",
            "draft_model_id": "Qwen/Qwen3.5-2B-Draft",
            "minimum_prompt_tokens": 4096,
            "keep_percentage": 25,
            "selection_chunck_token_count": 64,
            "mandatory_trailing_token_count": 256,
            "lookahead_token_count": 4,
            "importance_pooling_kernel_token_count": 9
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .speculative_prefill()
            .expect("speculative prefill config should resolve"),
        SpeculativePrefillConfig::new(
            true,
            Some("Qwen3.5-35B-Target".to_owned()),
            Some("Qwen/Qwen3.5-2B-Draft".to_owned()),
            4096,
            25,
            64,
            256,
            4,
            9,
        )
    );
}

#[test]
fn should_require_a_draft_model_when_speculative_prefill_is_enabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target"
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::SpeculativePrefillDraftModelRequired)
    ));
}

#[test]
fn should_require_a_target_model_when_speculative_prefill_is_enabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "enabled": true,
            "draft_model_id": "Qwen3.5-2B-Draft"
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::SpeculativePrefillTargetModelRequired)
    ));
}

#[test]
fn should_require_an_explicit_keep_percentage_when_speculative_prefill_is_enabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target",
            "draft_model_id": "Qwen/Qwen3.5-2B-Draft"
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageRequired)
    ));
}

#[test]
fn should_accept_the_minimum_and_maximum_explicit_keep_percentages() {
    for keep_percentage in [1_u32, 100_u32] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(
                r#"{{
                  "speculative_prefill": {{
                    "enabled": true,
                    "target_model_id": "Qwen3.5-35B-Target",
                    "draft_model_id": "Qwen/Qwen3.5-2B-Draft",
                    "keep_percentage": {keep_percentage}
                  }}
                }}"#,
            ),
        );

        let astronomical_config =
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
                .expect("a boundary keep percentage should load");
        assert_eq!(
            astronomical_config
                .speculative_prefill()
                .expect("speculative prefill config should resolve")
                .keep_percentage(),
            keep_percentage,
        );
    }
}

#[test]
fn should_reject_an_empty_speculative_prefill_target_model_id() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "enabled": false,
            "target_model_id": "   "
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::SpeculativePrefillTargetModelIdMustNotBeEmpty)
    ));
}

#[test]
fn should_reject_invalid_speculative_prefill_numeric_settings() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "draft_model_id": "Qwen/Qwen3.5-2B-Draft",
            "keep_percentage": 101
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageOutOfRange)
    ));
}

#[test]
fn should_allow_speculative_prefill_with_mtp_enabled_by_default() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target",
            "draft_model_id": "Qwen/Qwen3.5-2B-Draft",
            "keep_percentage": 20
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("MTP and speculative prefill should be composable");

    assert!(astronomical_config.mtp_enabled());
    assert!(
        astronomical_config
            .speculative_prefill()
            .expect("speculative prefill config should resolve")
            .is_enabled()
    );
}

#[test]
fn should_select_speculative_prefill_when_mtp_is_explicitly_disabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_enabled": false,
          "speculative_prefill": {
            "enabled": true,
            "target_model_id": "Qwen3.5-35B-Target",
            "draft_model_id": "Qwen/Qwen3.5-2B-Draft",
            "keep_percentage": 20
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.mtp_enabled());
    assert!(
        astronomical_config
            .speculative_prefill()
            .expect("speculative prefill config should resolve")
            .is_enabled()
    );
}

#[test]
fn should_allow_both_execution_modes_to_be_disabled() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "mtp_enabled": false,
          "speculative_prefill": { "enabled": false }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert!(!astronomical_config.mtp_enabled());
    assert!(
        !astronomical_config
            .speculative_prefill()
            .expect("speculative prefill config should resolve")
            .is_enabled()
    );
}
