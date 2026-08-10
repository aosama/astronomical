use super::*;

#[test]
fn should_resolve_every_user_configured_chunking_boundary() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "prefill_size_optimizer_enabled": true,
            "optimizer_prefill_token_candidates": [512, 1024],
            "full_attention_key_value_growth_tokens": 192,
            "speculative_prefill_draft_forward_tokens": 1536,
            "prefill_graph_submission_layer_interval": 0,
            "generation_graph_submission_layer_interval": 0,
            "prefill_optimizer_observation_window": 7,
            "prefill_optimizer_position_bucket_tokens": 16384,
            "prompt_cache_block_tokens": 1024,
            "prompt_cache_common_prefix_stride_blocks": 6
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("nested chunking configuration should load");
    let chunking = astronomical_config
        .chunking()
        .expect("chunking configuration should resolve");

    assert_eq!(chunking.full_attention_key_value_growth_tokens(), 192);
    assert_eq!(chunking.speculative_prefill_draft_forward_tokens(), 1_536);
    assert_eq!(chunking.prefill_graph_submission_layer_interval(), 0);
    assert_eq!(chunking.generation_graph_submission_layer_interval(), 0);
    assert_eq!(chunking.prefill_optimizer_observation_window(), 7);
    assert_eq!(chunking.prefill_optimizer_position_bucket_tokens(), 16_384);
    assert_eq!(chunking.prompt_cache_block_tokens(), Some(1_024));
    assert_eq!(chunking.prompt_cache_common_prefix_stride_blocks(), 6);
}

#[test]
fn should_reject_zero_for_chunking_boundaries_that_cannot_be_disabled() {
    for field_name in [
        "full_attention_key_value_growth_tokens",
        "speculative_prefill_draft_forward_tokens",
        "prefill_optimizer_observation_window",
        "prefill_optimizer_position_bucket_tokens",
        "prompt_cache_block_tokens",
        "prompt_cache_common_prefix_stride_blocks",
    ] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(
                r#"{{
                  "chunking": {{
                    "prefill_size_optimizer_enabled": true,
                    "{field_name}": 0
                  }}
                }}"#
            ),
        );

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::InvalidChunkingValue { .. })
        ));
    }
}

#[test]
fn should_reject_unknown_nested_chunking_fields() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "prefill_size_optimizer_enabled": true,
            "unrecognized_boundary": 512
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_reject_full_attention_growth_that_cannot_cross_the_mlx_dimension_boundary() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        &format!(
            r#"{{
              "chunking": {{
                "prefill_size_optimizer_enabled": true,
                "full_attention_key_value_growth_tokens": {}
              }}
            }}"#,
            u32::MAX
        ),
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::InvalidChunkingValue {
            field_name: "chunking.full_attention_key_value_growth_tokens",
            ..
        })
    ));
}

#[test]
fn should_reject_retired_gated_delta_dispatch_field() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "prefill_size_optimizer_enabled": true,
            "gated_delta_maximum_tokens_per_dispatch": 512
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_reject_retired_top_level_prefill_chunking_fields() {
    for retired_field_document in [
        r#"{ "prefill_chunck_size_optimizer_enabled": true }"#,
        r#"{ "fixed_prefill_chunck_tokens": 2048 }"#,
        r#"{ "optimizer_prefill_chunck_token_candidates": [1024, 2048] }"#,
    ] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(temporary_home_directory.path(), retired_field_document);

        assert!(matches!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
            Err(AstronomicalConfigError::ParseConfigFile { .. })
        ));
    }
}
