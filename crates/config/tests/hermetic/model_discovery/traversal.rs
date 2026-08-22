use std::fs;

use astronomical_config::{AstronomicalConfig, DiscoveredModelError};

use crate::hermetic::write_config;

use super::{write_minimal_model_config, write_required_model_files};

#[test]
fn should_resolve_model_directory_paths_from_the_configuration_array() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let config_directory = temporary_home_directory.path().join(".astronomical");
    let configured_model_directory = temporary_home_directory
        .path()
        .join("models/ConfiguredModel");
    fs::create_dir_all(&config_directory).expect("config directory should be created");
    fs::create_dir_all(&configured_model_directory).expect("model directory should be created");
    let config_json = serde_json::json!({
        "model_directories": [configured_model_directory],
        "chunking": {"fixed_prompt_processing_chunk_size_tokens": 2048}
    });
    fs::write(
        config_directory.join("config.json"),
        config_json.to_string(),
    )
    .expect("config should be written");

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(astronomical_config.model_directories().len(), 1);
    assert_eq!(
        astronomical_config.model_directories()[0],
        configured_model_directory
    );
}

#[test]
fn should_resolve_an_exact_model_directory_from_configured_recursive_roots() {
    let temporary_home_directory = tempfile::tempdir().expect("temporary home should be created");
    let configured_model_root = temporary_home_directory.path().join("models");
    let exact_model_directory = configured_model_root
        .join("nested")
        .join("ExactModel-OptiQ-4bit");
    fs::create_dir_all(&exact_model_directory).expect("model directory should be created");
    write_minimal_model_config(&exact_model_directory, "qwen3_5_moe", 262_144);
    write_required_model_files(&exact_model_directory);
    write_config(
        temporary_home_directory.path(),
        &serde_json::json!({
            "model_directories": [configured_model_root],
            "chunking": { "fixed_prompt_processing_chunk_size_tokens": 2048 },
        })
        .to_string(),
    );

    let astronomical_config =
        AstronomicalConfig::load_from_home_directory(temporary_home_directory.path())
            .expect("config should load");

    assert_eq!(
        astronomical_config
            .find_configured_model_directory_by_id("ExactModel-OptiQ-4bit")
            .expect("configured model discovery should complete"),
        Some(exact_model_directory)
    );
    assert_eq!(
        astronomical_config
            .find_configured_model_directory_by_id("MissingModel-OptiQ-4bit")
            .expect("missing model discovery should complete"),
        None
    );
}

#[test]
fn should_reject_duplicate_model_ids_with_deterministic_directory_order() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let first_root_directory = temporary_directory.path().join("root-z");
    let second_root_directory = temporary_directory.path().join("root-a");
    let first_model_directory = first_root_directory.join("SharedModel");
    let second_model_directory = second_root_directory.join("SharedModel");
    for model_directory in [&first_model_directory, &second_model_directory] {
        fs::create_dir_all(model_directory).expect("duplicate model directory should be created");
        write_minimal_model_config(model_directory, "qwen3_5_moe", 262_144);
        write_required_model_files(model_directory);
    }

    let discovery_error =
        astronomical_config::discover_models(&[first_root_directory, second_root_directory])
            .expect_err("duplicate model IDs should be rejected");

    let DiscoveredModelError::DuplicateModelId {
        model_id,
        model_directories,
    } = discovery_error
    else {
        panic!("duplicate discovery must return DuplicateModelId");
    };
    let mut expected_directories = vec![first_model_directory, second_model_directory];
    expected_directories.sort();
    assert_eq!(model_id, "SharedModel");
    assert_eq!(model_directories, expected_directories);
}

#[test]
fn should_discover_an_organization_model_tree_and_skip_hidden_incomplete_staging() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let models_root_directory = temporary_directory.path().join("models");
    let published_model_directory = models_root_directory
        .join("astronomical-test")
        .join("example-qwen");
    let incomplete_model_directory = models_root_directory
        .join(".incomplete")
        .join("astronomical-test")
        .join("staged-qwen");
    for model_directory in [&published_model_directory, &incomplete_model_directory] {
        fs::create_dir_all(model_directory).expect("model fixture directory should be created");
        write_minimal_model_config(model_directory, "qwen3_5_moe", 262_144);
        write_required_model_files(model_directory);
    }

    let directory_scans = astronomical_config::discover_models(&[models_root_directory])
        .expect("ordinary model discovery should complete");
    let discovered_models = &directory_scans[0].discovered_models;

    assert_eq!(discovered_models.len(), 1);
    assert_eq!(discovered_models[0].model_id, "example-qwen");
    assert_eq!(
        discovered_models[0].model_directory,
        published_model_directory
    );
    let shard_size = fs::metadata(
        discovered_models[0]
            .model_directory
            .join("model-00001.safetensors"),
    )
    .expect("the published model shard metadata should be readable")
    .len();
    assert!(shard_size > 0);
    assert_eq!(discovered_models[0].model_size_bytes, shard_size);
}
