use super::*;

#[test]
fn should_require_fixed_prefill_chunck_tokens_when_optimizer_is_disabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "prefill_chunck_size_optimizer_enabled": false
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::FixedPrefillChunckTokensRequiredWhenOptimizerDisabled)
    ));
}

#[test]
fn should_reject_zero_fixed_prefill_chunck_tokens_when_optimizer_is_disabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "prefill_chunck_size_optimizer_enabled": false,
          "fixed_prefill_chunck_tokens": 0
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::InvalidFixedPrefillChunckTokens)
    ));
}

#[test]
fn should_accept_and_ignore_fixed_prefill_chunck_tokens_when_optimizer_is_enabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "prefill_chunck_size_optimizer_enabled": true,
          "fixed_prefill_chunck_tokens": 4096,
          "optimizer_prefill_chunck_token_candidates": [1024, 2048, 4096, 8192]
        }"#,
    );

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("config should load with optimizer enabled and fixed tokens ignored");

    assert_eq!(
        user_config
            .prefill_chunck_sizing_policy()
            .expect("the optimizer policy should resolve"),
        PrefillChunckSizingPolicy::Optimized {
            optimizer_prefill_chunck_token_candidates: vec![1_024, 2_048, 4_096, 8_192],
        }
    );
    assert_eq!(
        user_config.ignored_fixed_prefill_chunck_tokens(),
        Some(4_096),
        "the ignored fixed token count should be surfaced so the menu can warn the user"
    );
}

#[test]
fn should_reject_an_empty_optimizer_prefill_chunck_candidate_array() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{
          "prefill_chunck_size_optimizer_enabled": true,
          "optimizer_prefill_chunck_token_candidates": []
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temp_home.path()),
        Err(AstronomicalConfigError::OptimizerPrefillChunckTokenCandidatesMustNotBeEmpty)
    ));
}

#[test]
fn should_not_report_ignored_fixed_prefill_chunck_tokens_when_only_optimizer_is_enabled() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{ "prefill_chunck_size_optimizer_enabled": true }"#,
    );

    let user_config =
        AstronomicalConfig::load_from_home_directory(temp_home.path()).expect("config should load");

    assert_eq!(
        user_config.ignored_fixed_prefill_chunck_tokens(),
        None,
        "no ignore warning should be reported when fixed tokens are not configured"
    );
}

#[test]
fn should_ignore_fixed_prefill_chunck_tokens_when_optimizer_setting_is_omitted() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temp_home.path(),
        r#"{ "fixed_prefill_chunck_tokens": 4096 }"#,
    );

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("config should default to optimized prompt processing");

    assert_eq!(
        user_config
            .prefill_chunck_sizing_policy()
            .expect("the optimized prefill policy should resolve"),
        PrefillChunckSizingPolicy::Optimized {
            optimizer_prefill_chunck_token_candidates: vec![1_024, 2_048, 4_096, 8_192],
        }
    );
    assert_eq!(
        user_config.ignored_fixed_prefill_chunck_tokens(),
        None,
        "only an explicit optimizer setting reports a fixed-size override warning"
    );
}

#[test]
fn should_default_to_optimized_prefill_when_optimizer_setting_is_omitted() {
    let temp_home = tempfile::tempdir().expect("temp home should be created");
    let config_directory = temp_home.path().join(".astronomical");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), "{}")
        .expect("config file should be written");

    let user_config = AstronomicalConfig::load_from_home_directory(temp_home.path())
        .expect("omitted sizing settings should select the optimizer");

    assert_eq!(
        user_config
            .prefill_chunck_sizing_policy()
            .expect("optimized policy should resolve"),
        PrefillChunckSizingPolicy::Optimized {
            optimizer_prefill_chunck_token_candidates: vec![1_024, 2_048, 4_096, 8_192],
        }
    );
}

#[test]
fn should_reject_the_retired_prefill_chunck_tokens_config_field() {
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
            AstronomicalConfigError::OptimizerPrefillChunckTokenCandidatesMustBePositive,
        ),
        (
            "[1024, 1024]",
            AstronomicalConfigError::OptimizerPrefillChunckTokenCandidatesMustBeStrictlyIncreasing,
        ),
        (
            "[2048, 1024]",
            AstronomicalConfigError::OptimizerPrefillChunckTokenCandidatesMustBeStrictlyIncreasing,
        ),
    ] {
        let temp_home = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temp_home.path(),
            &format!(
                r#"{{
                  "prefill_chunck_size_optimizer_enabled": true,
                  "optimizer_prefill_chunck_token_candidates": {configured_candidates}
                }}"#
            ),
        );

        let config_error = AstronomicalConfig::load_from_home_directory(temp_home.path())
            .expect_err("invalid optimizer candidates should fail config loading");
        assert_eq!(config_error.to_string(), expected_error.to_string());
    }
}
