//! Config-file resolution coverage used by the reload journey.

use std::path::PathBuf;

use astronomical_supervisor::ResolvedRuntimeConfigResolver;

#[test]
fn should_resolve_reload_config_from_the_config_file() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let config_file_path = config_home_directory
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": {
                "model_directories": []
            },
            "chunking": {
                "fixed_prompt_processing_chunk_size_tokens": 4096,
                "full_attention_key_value_growth_tokens": 192,
                "speculative_prefill_draft_forward_tokens": 1536,
                "prefill_graph_submission_layer_interval": 1,
                "experimental_ssd_paging_generation_graph_submission_layer_interval": 6,
                "prompt_cache_block_tokens": 768,
                "prompt_cache_common_prefix_stride_blocks": 8
            },
            "prompt_cache": {
                "maximum_size_gb": 1
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver.load().expect("the reload config should resolve");

    assert_eq!(resolved_config.bind_address, "127.0.0.1:6733");
    assert_eq!(
        resolved_config.worker_executable_path,
        PathBuf::from("/fallback/worker")
    );
    assert_eq!(
        resolved_config
            .prompt_cache_config
            .global_prompt_cache_root_directory(),
        &config_home_directory.join(".astronomical-dev/cache")
    );
}

#[test]
fn should_not_resolve_a_draft_model_when_speculative_prefill_is_disabled() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let config_file_path = config_home_directory
        .path()
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "speculative_prefill": {
                "enabled": false,
                "draft_model_id": "astronomical/unused-draft"
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_runtime_config = resolver
        .load()
        .expect("a disabled speculative-prefill draft must not require discovery");

    assert!(resolved_runtime_config.model_policy_catalog.is_empty());
}

#[test]
fn should_preserve_speculative_prefill_override_when_target_model_is_not_discovered() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let config_file_path = config_home_directory
        .path()
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": {"model_directories": []},
            "models": {
                "target-model": {
                    "acceleration": {
                        "speculative_prefill": {
                            "draft_model_id": "draft-model",
                            "keep_percentage": 20
                        }
                    }
                }
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver
        .load()
        .expect("a dormant override must not prevent ordinary serving");
    assert_eq!(
        resolved_config.unmatched_model_config_ids,
        vec!["target-model".to_owned()]
    );
    assert!(resolved_config.model_policy_catalog.is_empty());
}
