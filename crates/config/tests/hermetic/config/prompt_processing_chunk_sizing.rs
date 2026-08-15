use super::*;

#[test]
fn should_require_fixed_prompt_processing_chunk_size_when_optimizer_is_disabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": { "prompt_processing_chunk_size_optimizer_enabled": false }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::FixedPromptProcessingChunkSizeTokensRequiredWhenOptimizerDisabled)
    ));
}

#[test]
fn should_accept_fixed_ssd_streaming_prompt_processing_chunk_size_when_optimizer_is_disabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": false,
            "fixed_prompt_processing_chunk_size_tokens": 2048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 256
          }
        }"#,
    );

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("config should load with fixed SSD streaming prefill tokens");

    assert_eq!(
        user_config
            .prompt_processing_chunk_sizing_policy()
            .expect("the fixed policy should resolve"),
        PromptProcessingChunkSizingPolicy::Fixed {
            fixed_prompt_processing_chunk_size_tokens: 2_048,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Some(256),
        }
    );
}

#[test]
fn should_reject_zero_fixed_ssd_streaming_prompt_processing_chunk_size() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": false,
            "fixed_prompt_processing_chunk_size_tokens": 2048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 0
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::InvalidFixedSsdStreamingPromptProcessingChunkSizeTokens)
    ));
}

#[test]
fn should_reject_zero_fixed_prompt_processing_chunk_size_when_optimizer_is_disabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": false,
            "fixed_prompt_processing_chunk_size_tokens": 0
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::InvalidFixedPromptProcessingChunkSizeTokens)
    ));
}

#[test]
fn should_accept_and_ignore_fixed_chunk_size_when_optimizer_is_enabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": true,
            "fixed_prompt_processing_chunk_size_tokens": 4096,
            "prompt_processing_chunk_size_optimizer_candidate_token_counts": [1024, 2048, 4096, 8192]
          }
        }"#,
    );

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("config should load with optimizer enabled and fixed tokens ignored");

    assert_eq!(
        user_config
            .prompt_processing_chunk_sizing_policy()
            .expect("the optimizer policy should resolve"),
        PromptProcessingChunkSizingPolicy::Optimized {
            prompt_processing_chunk_size_optimizer_candidate_token_counts: vec![
                1_024, 2_048, 4_096, 8_192
            ],
        }
    );
    assert_eq!(
        user_config.ignored_fixed_prompt_processing_chunk_size_tokens(),
        Some(4_096),
        "the ignored fixed token count should be surfaced so the menu can warn the user"
    );
}

#[test]
fn should_reject_an_empty_optimizer_candidate_array() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": true,
            "prompt_processing_chunk_size_optimizer_candidate_token_counts": []
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::OptimizerCandidateTokenCountsMustNotBeEmpty)
    ));
}

#[test]
fn should_not_report_ignored_fixed_size_when_only_optimizer_is_enabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{ "chunking": { "prompt_processing_chunk_size_optimizer_enabled": true } }"#,
    );

    let user_config =
        AstronomicalConfig::load_from_home_directory(temp_home.path()).expect("config should load");

    assert_eq!(
        user_config.ignored_fixed_prompt_processing_chunk_size_tokens(),
        None,
        "no ignore warning should be reported when fixed tokens are not configured"
    );
}

#[test]
fn should_ignore_fixed_chunk_size_when_optimizer_setting_is_omitted() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{ "chunking": { "fixed_prompt_processing_chunk_size_tokens": 4096 } }"#,
    );

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("config should default to optimized prompt processing");

    assert_eq!(
        user_config
            .prompt_processing_chunk_sizing_policy()
            .expect("the optimized prefill policy should resolve"),
        PromptProcessingChunkSizingPolicy::Optimized {
            prompt_processing_chunk_size_optimizer_candidate_token_counts: vec![
                1_024, 2_048, 4_096, 8_192
            ],
        }
    );
    assert_eq!(
        user_config.ignored_fixed_prompt_processing_chunk_size_tokens(),
        None,
        "only an explicit optimizer setting reports a fixed-size override warning"
    );
}

#[test]
fn should_default_to_optimized_prompt_processing_when_setting_is_omitted() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    let config_directory = temp_home.path().join(".astronomical");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), "{}")
        .expect("config file should be written");

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("omitted sizing settings should select the optimizer");

    assert_eq!(
        user_config
            .prompt_processing_chunk_sizing_policy()
            .expect("optimized policy should resolve"),
        PromptProcessingChunkSizingPolicy::Optimized {
            prompt_processing_chunk_size_optimizer_candidate_token_counts: vec![
                1_024, 2_048, 4_096, 8_192
            ],
        }
    );
}

#[test]
fn should_reject_the_retired_top_level_chunk_size_field() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "prefill_chunck_tokens": 2048
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_reject_zero_duplicate_and_descending_optimizer_candidates() {
    for (configured_candidates, expected_error) in [
        (
            "[0, 1024]",
            AstronomicalConfigError::OptimizerCandidateTokenCountsMustBePositive,
        ),
        (
            "[1024, 1024]",
            AstronomicalConfigError::OptimizerCandidateTokenCountsMustBeStrictlyIncreasing,
        ),
        (
            "[2048, 1024]",
            AstronomicalConfigError::OptimizerCandidateTokenCountsMustBeStrictlyIncreasing,
        ),
    ] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(
                r#"{{
              "chunking": {{
                "prompt_processing_chunk_size_optimizer_enabled": true,
                "prompt_processing_chunk_size_optimizer_candidate_token_counts": {configured_candidates}
              }}
                }}"#
            ),
        );

        let config_error = AstronomicalConfig::load_from_home_directory(temp_home.path())
            .expect_err("invalid optimizer candidates should fail config loading");
        assert_eq!(config_error.to_string(), expected_error.to_string());
    }
}

#[test]
fn should_reject_every_retired_optimizer_configuration_key() {
    for retired_key in [
        "prefill_size_optimizer_enabled",
        "fixed_prefill_tokens",
        "fixed_ssd_streaming_prefill_tokens",
        "optimizer_prefill_token_candidates",
        "prefill_optimizer_observation_window",
        "prefill_optimizer_position_bucket_tokens",
    ] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(r#"{{ "chunking": {{ "{retired_key}": true }} }}"#),
        );

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temp_home.path()),
            Err(AstronomicalConfigError::ParseConfigFile { .. })
        ));
    }
}
