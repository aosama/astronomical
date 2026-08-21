//! Config-file and model-root resolution coverage used by the reload journey.

use std::fs;
use std::path::{Path, PathBuf};

use astronomical_config::DiscoveredModelError;
use astronomical_supervisor::{ResolvedRuntimeConfigError, ResolvedRuntimeConfigResolver};

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

#[test]
fn should_treat_a_missing_automatic_models_directory_as_an_empty_library_without_creating_it() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    write_development_config(config_home_directory.path(), &[]);
    let resolver = development_resolver(config_home_directory.path());
    let automatic_models_directory = resolver.instance_paths().models_directory();

    let resolved_config = resolver
        .load()
        .expect("an absent automatic library should resolve as empty");

    assert!(resolved_config.discovered_models.is_empty());
    assert!(resolved_config.configured_model_directories.is_empty());
    assert!(!automatic_models_directory.exists());
}

#[test]
fn should_discover_the_automatic_organization_model_tree_before_configured_roots() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let automatic_models_directory = config_home_directory
        .path()
        .join(".astronomical-dev/models");
    let automatic_model_directory = automatic_models_directory
        .join("astronomical-test")
        .join("example-qwen");
    let configured_models_directory = config_home_directory.path().join("configured-models");
    let configured_model_directory = configured_models_directory
        .join("astronomical-test")
        .join("configured-qwen");
    write_minimal_qwen_model(&automatic_model_directory);
    write_minimal_qwen_model(&configured_model_directory);
    write_development_config(
        config_home_directory.path(),
        std::slice::from_ref(&configured_models_directory),
    );
    let resolver = development_resolver(config_home_directory.path());

    let resolved_config = resolver
        .load()
        .expect("automatic and configured libraries should resolve together");
    let discovered_model_ids = resolved_config
        .discovered_models
        .iter()
        .map(|model| model.model_id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(discovered_model_ids, ["example-qwen", "configured-qwen"]);
    assert!(
        resolved_config
            .discovered_models
            .iter()
            .all(|discovered_model| discovered_model.model_size_bytes > 0)
    );
    assert_eq!(
        resolved_config.configured_model_directories,
        vec![configured_models_directory]
    );
}

#[test]
fn should_scan_a_repeated_automatic_root_once_while_preserving_authored_configuration() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let automatic_models_directory = config_home_directory
        .path()
        .join(".astronomical-dev/models");
    write_minimal_qwen_model(
        &automatic_models_directory
            .join("astronomical-test")
            .join("example-qwen"),
    );
    write_development_config(
        config_home_directory.path(),
        std::slice::from_ref(&automatic_models_directory),
    );
    let resolver = development_resolver(config_home_directory.path());

    let resolved_config = resolver
        .load()
        .expect("a repeated lexical root should not duplicate model discovery");

    assert_eq!(resolved_config.discovered_models.len(), 1);
    assert_eq!(
        resolved_config.configured_model_directories,
        vec![automatic_models_directory]
    );
}

#[test]
fn should_retain_the_configured_root_failure_when_the_missing_automatic_path_is_authored() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let missing_models_directory = config_home_directory
        .path()
        .join(".astronomical-dev/models");
    write_development_config(
        config_home_directory.path(),
        std::slice::from_ref(&missing_models_directory),
    );
    let resolver = development_resolver(config_home_directory.path());

    let resolution_error = resolver
        .load()
        .expect_err("an authored missing root should retain its discovery failure");

    assert!(matches!(
        resolution_error,
        ResolvedRuntimeConfigError::ModelDiscovery(DiscoveredModelError::ReadDirectory {
            directory_path,
            ..
        }) if directory_path == missing_models_directory
    ));
}

#[test]
fn should_fail_closed_when_the_existing_automatic_models_path_is_not_a_directory() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    write_development_config(config_home_directory.path(), &[]);
    let resolver = development_resolver(config_home_directory.path());
    let automatic_models_directory = resolver.instance_paths().models_directory();
    fs::write(&automatic_models_directory, b"not a directory")
        .expect("the non-directory fixture should be written");

    let resolution_error = resolver
        .load()
        .expect_err("an invalid automatic root should fail closed");

    assert!(matches!(
        resolution_error,
        ResolvedRuntimeConfigError::ModelDiscovery(DiscoveredModelError::ReadDirectory {
            directory_path,
            ..
        }) if directory_path == automatic_models_directory
    ));
}

#[cfg(unix)]
#[test]
fn should_return_a_typed_error_when_automatic_root_metadata_cannot_be_read() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    write_development_config(config_home_directory.path(), &[]);
    let resolver = development_resolver(config_home_directory.path());
    let automatic_models_directory = resolver.instance_paths().models_directory();
    std::os::unix::fs::symlink("models", &automatic_models_directory)
        .expect("the automatic-root symlink loop should be created");

    let resolution_error = resolver
        .load()
        .expect_err("automatic-root metadata failure should remain typed");

    assert!(matches!(
        resolution_error,
        ResolvedRuntimeConfigError::AutomaticModelDirectoryMetadata {
            model_directory,
            ..
        } if model_directory == automatic_models_directory
    ));
}

#[test]
fn should_reject_duplicate_model_identities_from_distinct_effective_roots() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let automatic_models_directory = config_home_directory
        .path()
        .join(".astronomical-dev/models");
    let automatic_model_directory = automatic_models_directory
        .join("astronomical-test")
        .join("shared-qwen");
    let configured_models_directory = config_home_directory.path().join("configured-models");
    let configured_model_directory = configured_models_directory
        .join("another-organization")
        .join("shared-qwen");
    write_minimal_qwen_model(&automatic_model_directory);
    write_minimal_qwen_model(&configured_model_directory);
    write_development_config(
        config_home_directory.path(),
        std::slice::from_ref(&configured_models_directory),
    );
    let resolver = development_resolver(config_home_directory.path());

    let resolution_error = resolver
        .load()
        .expect_err("distinct roots with one public identity should remain ambiguous");

    let ResolvedRuntimeConfigError::ModelDiscovery(DiscoveredModelError::DuplicateModelId {
        model_id,
        model_directories,
    }) = resolution_error
    else {
        panic!("duplicate identities should return the typed discovery error");
    };
    let mut expected_model_directories =
        vec![automatic_model_directory, configured_model_directory];
    expected_model_directories.sort();
    assert_eq!(model_id, "shared-qwen");
    assert_eq!(model_directories, expected_model_directories);
}

fn development_resolver(config_home_directory: &Path) -> ResolvedRuntimeConfigResolver {
    ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.to_path_buf(),
        PathBuf::from("/fictional/fallback-worker"),
    )
}

fn write_development_config(config_home_directory: &Path, model_directories: &[PathBuf]) {
    let config_file_path = config_home_directory
        .join(".astronomical-dev")
        .join("config.json");
    fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    let config_document = serde_json::json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {"model_directories": model_directories},
    });
    fs::write(
        config_file_path,
        serde_json::to_vec(&config_document).expect("the config fixture should serialize"),
    )
    .expect("the config fixture should be written");
}

fn write_minimal_qwen_model(model_directory: &Path) {
    const MODEL_SHARD_BYTES: &[u8] = b"fictional-shard";
    fs::create_dir_all(model_directory).expect("the model fixture directory should be created");
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":262144}}"#,
    )
    .expect("the model config should be written");
    fs::write(
        model_directory.join("model-00001.safetensors"),
        MODEL_SHARD_BYTES,
    )
    .expect("the model shard should be written");
    fs::write(
        model_directory.join("model.safetensors.index.json"),
        format!(
            r#"{{"metadata":{{"total_size":{}}},"weight_map":{{"model.embed_tokens.weight":"model-00001.safetensors"}}}}"#,
            MODEL_SHARD_BYTES.len()
        ),
    )
    .expect("the model index should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":1,"model":{"type":"BPE"}}"#,
    )
    .expect("the tokenizer fixture should be written");
}
