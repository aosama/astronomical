use super::*;

fn persisted_chunking_object(home_directory: &std::path::Path) -> serde_json::Value {
    let persisted_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(home_directory.join(".astronomical/config.json"))
            .expect("the rewritten config should be readable"),
    )
    .expect("the rewritten config should be JSON");
    persisted_config["chunking"].clone()
}

#[test]
fn should_resolve_every_user_configured_chunking_boundary() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_tokens": 1024,
            "full_attention_key_value_growth_tokens": 192,
            "prefill_graph_submission_layer_interval": 2,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 0,
            "ssd_prompt_cache_block_tokens": 1024,
            "ssd_prompt_cache_common_prefix_stride_blocks": 6
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
    assert_eq!(chunking.prefill_graph_submission_layer_interval(), 2);
    assert_eq!(
        chunking.experimental_ssd_paging_generation_graph_submission_layer_interval(),
        0
    );
    assert_eq!(chunking.prompt_processing_chunk_size_tokens(), Some(1_024));
    assert_eq!(chunking.ssd_prompt_cache_block_tokens(), Some(1_024));
    assert_eq!(chunking.ssd_prompt_cache_common_prefix_stride_blocks(), 6);

    let persisted_chunking = persisted_chunking_object(temporary_home_directory.path());
    assert_eq!(
        persisted_chunking["prefill_graph_submission_layer_interval"],
        2
    );
}

#[test]
fn should_default_omitted_graph_submission_intervals() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {}
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("omitted experimental solid-state-drive paging intervals should load");
    let chunking = astronomical_config
        .chunking()
        .expect("chunking configuration should resolve");

    assert_eq!(chunking.prefill_graph_submission_layer_interval(), 0);
    assert_eq!(
        chunking.experimental_ssd_paging_generation_graph_submission_layer_interval(),
        3
    );
}

#[test]
fn should_default_the_decode_levers_off_so_the_measured_fast_path_serves() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {}
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("omitted experimental decode levers should load");
    let chunking = astronomical_config
        .chunking()
        .expect("chunking configuration should resolve");

    assert!(
        !chunking.experimental_neural_engine_prefill_enabled(),
        "Neural Engine prefill holds about twice its programs' weights in memory and must stay opt-in"
    );
}

#[test]
fn should_load_a_saved_file_that_still_names_the_retired_ssd_prefill_interval() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "prompt_processing_chunk_size_tokens": 2048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2048,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 3
          }
        }"#,
    );

    let chunking = AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("a file saved before the setting was retired should load")
        .chunking()
        .expect("chunking should resolve");

    assert_eq!(chunking.prefill_graph_submission_layer_interval(), 0);
    assert_eq!(
        chunking.experimental_ssd_paging_generation_graph_submission_layer_interval(),
        3
    );
    let persisted_chunking = persisted_chunking_object(temporary_home_directory.path());
    assert!(
        persisted_chunking
            .get("experimental_ssd_paging_prefill_graph_submission_layer_interval")
            .is_none(),
        "the retired setting must be removed from the saved file"
    );
    assert_eq!(
        persisted_chunking["prefill_graph_submission_layer_interval"],
        0
    );
    // 2,048 is the retired SSD-paged default, so load carries it forward.
    assert_eq!(
        persisted_chunking["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"],
        config::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS
    );
    assert_eq!(
        persisted_chunking["experimental_ssd_paging_generation_graph_submission_layer_interval"],
        3
    );
}

#[test]
fn should_reject_zero_for_chunking_boundaries_that_cannot_be_disabled() {
    for field_name in [
        "full_attention_key_value_growth_tokens",
        "ssd_prompt_cache_block_tokens",
        "ssd_prompt_cache_common_prefix_stride_blocks",
    ] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(
                r#"{{
                  "chunking": {{
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

fn drafted_tokens_configured_as(
    draft_tokens_json: &str,
) -> Result<Option<u32>, AstronomicalConfigError> {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        &format!(
            r#"{{"chunking": {{"multi_token_prediction_draft_tokens": {draft_tokens_json}}}}}"#
        ),
    );
    AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())?
        .chunking()
        .map(|chunking| chunking.multi_token_prediction_draft_tokens())
}

#[test]
fn should_leave_the_draft_count_to_the_engine_when_it_is_null_or_absent() {
    assert_eq!(
        drafted_tokens_configured_as("null").expect("null loads"),
        None
    );
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(temporary_home_directory.path(), r#"{"chunking": {}}"#);
    let chunking = AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
        .expect("an empty chunking object loads")
        .chunking()
        .expect("chunking resolves");
    assert_eq!(chunking.multi_token_prediction_draft_tokens(), None);
}

#[test]
fn should_accept_no_drafting_and_the_largest_draft_count() {
    assert_eq!(drafted_tokens_configured_as("0").expect("0 loads"), Some(0));
    assert_eq!(drafted_tokens_configured_as("3").expect("3 loads"), Some(3));
}

#[test]
fn should_reject_a_draft_count_beyond_what_a_head_drafts() {
    let refusal = drafted_tokens_configured_as("4").expect_err("4 drafted tokens are refused");
    assert!(
        refusal
            .to_string()
            .contains("multi_token_prediction_draft_tokens"),
        "the refusal names the setting: {refusal}"
    );
}

/// A file saved by 0.2.90 can hold these, and refusing it would stop the daemon on update.
#[test]
fn should_load_a_config_holding_settings_retired_after_0_2_90() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "chunking": {
            "prompt_processing_chunk_size_tokens": 6144,
            "experimental_quantized_kv_cache_enabled": true,
            "experimental_fused_moe_decode_enabled": true
          },
          "models": {
            "target": {
              "chunking": {
                "full_attention_key_value_growth_tokens": 512,
                "experimental_quantized_kv_cache_enabled": false,
                "experimental_fused_moe_decode_enabled": false
              }
            }
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("retired settings must never stop a config from loading");

    let model_config = astronomical_config
        .resolved_model_config("target", 65_536)
        .expect("the model policy should resolve");
    assert_eq!(
        model_config
            .chunking()
            .prompt_processing_chunk_size_tokens(),
        Some(6144),
        "the settings beside the retired ones must survive"
    );
    assert_eq!(
        model_config
            .chunking()
            .full_attention_key_value_growth_tokens(),
        512
    );
    // Rewriting the file is what stops the warning repeating on every load.
    let rewritten_config = std::fs::read_to_string(
        temporary_home_directory
            .path()
            .join(".astronomical/config.json"),
    )
    .expect("the rewritten config should be readable");
    assert!(!rewritten_config.contains("experimental_quantized_kv_cache_enabled"));
    assert!(!rewritten_config.contains("experimental_fused_moe_decode_enabled"));
}

/// The two stage attribution settings no engine read (issue #2579) are
/// dropped wherever a settings file still holds them, so a file written by
/// an earlier version still loads and is written back without them.
#[test]
fn should_load_a_config_holding_the_retired_stage_attribution_settings() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "chunking": {
            "prompt_processing_chunk_size_tokens": 6144,
            "experimental_decode_stage_attribution_enabled": true
          },
          "models": {
            "target": {
              "chunking": {"experimental_prefill_stage_attribution_enabled": true}
            }
          }
        }"#,
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("retired stage attribution settings must never stop a config from loading");

    assert_eq!(
        astronomical_config
            .resolved_model_config("target", 65_536)
            .expect("the model policy should resolve")
            .chunking()
            .prompt_processing_chunk_size_tokens(),
        Some(6144),
        "the setting beside a retired one must survive"
    );
    let rewritten_config = std::fs::read_to_string(
        temporary_home_directory
            .path()
            .join(".astronomical/config.json"),
    )
    .expect("the rewritten config should be readable");
    assert!(!rewritten_config.contains("stage_attribution"));
}

#[test]
fn should_load_a_config_holding_retired_settings_in_only_one_model() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "models": {
            "first": {"chunking": {"experimental_fused_moe_decode_enabled": true}},
            "second": {}
          }
        }"#,
    );

    assert!(AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()).is_ok());
}

#[test]
fn should_still_reject_an_unknown_chunking_field_beside_a_retired_one() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": {"model_directories": []},
          "chunking": {
            "experimental_quantized_kv_cache_enabled": true,
            "experimental_quantized_kv_cache_enable": true
          }
        }"#,
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_migrate_a_legacy_config_holding_settings_retired_after_0_2_90() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "model_directories": [],
          "maximum_mlx_memory_gb": 16,
          "chunking": {
            "experimental_quantized_kv_cache_enabled": true,
            "experimental_fused_moe_decode_enabled": true
          }
        }"#,
    );

    assert!(AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()).is_ok());
}

#[test]
fn should_reject_unknown_nested_chunking_fields() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
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
fn should_reject_retired_graph_submission_layer_interval_fields() {
    let retired_field_name = "generation_graph_submission_layer_interval";
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        &format!(
            r#"{{
              "chunking": {{
                "{retired_field_name}": 1
              }}
            }}"#
        ),
    );

    assert!(matches!(
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()),
        Err(AstronomicalConfigError::ParseConfigFile { .. })
    ));
}

#[test]
fn should_reject_retired_gated_delta_dispatch_field() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
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

#[test]
fn should_accept_and_ignore_the_speculative_decoding_block_length() {
    let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
    write_config(
        temporary_home_directory.path(),
        r#"{
          "chunking": {
            "speculative_decoding_block_length_tokens": 4
          }
        }"#,
    );

    // DFlash speculative decoding is disabled by design: the key parses
    // (a configuration written while it was enabled must keep loading)
    // and the rest of the chunking policy still resolves to its defaults.
    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("the configuration carrying the ignored speculative key should load");
    let chunking = astronomical_config
        .chunking()
        .expect("chunking configuration should resolve");

    assert_eq!(chunking.ssd_prompt_cache_common_prefix_stride_blocks(), 4);
}

#[test]
fn should_ignore_out_of_range_speculative_decoding_block_lengths() {
    for ignored_block_length in [0_usize, 65] {
        let temporary_home_directory = tempfile::tempdir().expect("temp home should be created");
        write_config(
            temporary_home_directory.path(),
            &format!(
                r#"{{
                  "chunking": {{
                    "speculative_decoding_block_length_tokens": {}
                  }}
                }}"#,
                ignored_block_length
            ),
        );

        assert!(
            AstronomicalConfig::load_from_home_directory(temporary_home_directory.path()).is_ok(),
            "out-of-range speculative block lengths are ignored, not rejected: \
             the key carries no policy while DFlash is disabled by design"
        );
    }
}

#[test]
fn should_let_one_model_opt_into_neural_engine_prefill_without_changing_the_others() {
    let temporary_home = tempfile::tempdir().expect("a temporary home directory");
    let config_directory = temporary_home.path().join(".astronomical-dev");
    std::fs::create_dir_all(&config_directory).expect("the config directory is created");
    std::fs::write(
        config_directory.join("config.json"),
        r#"{
          "$schema": "./astronomical-config.schema.json",
          "schema_version": 1,
          "runtime": { "model_directories": [] },
          "models": {
            "opted-in-model": {
              "chunking": { "experimental_neural_engine_prefill_enabled": true }
            }
          }
        }"#,
    )
    .expect("the config is written");

    let config = AstronomicalConfig::load_from_development_home_directory(temporary_home.path())
        .expect("the configuration loads");

    let opted_in_policy = config
        .resolved_model_config("opted-in-model", 262_144)
        .expect("the model policy resolves");
    let other_policy = config
        .resolved_model_config("other-model", 262_144)
        .expect("the model policy resolves");
    assert!(
        opted_in_policy
            .chunking()
            .experimental_neural_engine_prefill_enabled()
    );
    assert!(
        !other_policy
            .chunking()
            .experimental_neural_engine_prefill_enabled()
    );
}

/// A new user starts with no configuration file at all: the first load writes
/// one. The Neural Engine prefill must be off in what that load resolves, for
/// every model, and the written file must not switch it on, because the next
/// load reads that file. Both instances share the first-run path.
#[test]
fn should_leave_neural_engine_prefill_off_on_a_fresh_install() {
    for load_instance in [
        AstronomicalConfig::load_from_home_directory
            as fn(PathBuf) -> Result<AstronomicalConfig, AstronomicalConfigError>,
        AstronomicalConfig::load_from_development_home_directory,
    ] {
        let fresh_home = tempfile::tempdir().expect("an empty home directory");

        let first_load = load_instance(fresh_home.path().to_path_buf())
            .expect("a fresh install writes and loads its first-run configuration");
        let written_config_path = first_load.instance_paths().config_file_path();
        let written_config: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&written_config_path).expect("the first-run file was written"),
        )
        .expect("the first-run file is JSON");
        let second_load = load_instance(fresh_home.path().to_path_buf())
            .expect("the written first-run configuration loads again");

        for loaded_config in [&first_load, &second_load] {
            assert!(
                !loaded_config
                    .chunking()
                    .expect("chunking resolves")
                    .experimental_neural_engine_prefill_enabled(),
                "a fresh install must not run prefill on the Neural Engine"
            );
            assert!(
                !loaded_config
                    .resolved_model_config("any-model", 262_144)
                    .expect("a model's policy resolves")
                    .chunking()
                    .experimental_neural_engine_prefill_enabled(),
                "a model with no entry of its own must inherit the off default"
            );
        }
        assert_ne!(
            written_config["chunking"]["experimental_neural_engine_prefill_enabled"],
            serde_json::json!(true),
            "the first-run file must not opt the user in"
        );
    }
}
