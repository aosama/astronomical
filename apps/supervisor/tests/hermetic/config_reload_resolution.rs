//! MTP pairing resolution and speculative-prefill discovery integration tests.
//!
//! These tests verify that supervisor resolution discovers auxiliary MTP
//! drafters, preserves missing drafters as unavailable pairings, rejects
//! targets that are not publicly discovered, and disables speculative-prefill
//! draft discovery when speculative prefill is disabled.

use std::path::PathBuf;

use astronomical_supervisor::{ResolvedRuntimeConfigError, ResolvedRuntimeConfigResolver};

#[test]
fn should_resolve_target_and_auxiliary_mtp_pairing_without_advertising_the_drafter() {
    let config_home_directory = tempfile::tempdir().expect("config home should be created");
    let model_root_directory = config_home_directory.path().join("models");
    let target_model_directory = model_root_directory.join("Target-Model");
    let drafter_model_directory = model_root_directory.join("Target-Model-MTP");
    write_shallow_target_model(&target_model_directory);
    write_shallow_mtp_drafter(&drafter_model_directory);
    write_development_config(
        config_home_directory.path(),
        &format!(
            r#"{{
              "model_directories": [{}],
              "mtp_pairings": [{{
                "target_model_id": "Target-Model",
                "drafter_model_id": "Target-Model-MTP"
              }}]
            }}"#,
            serde_json::to_string(&model_root_directory).expect("model root should serialize")
        ),
    );
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver.load().expect("pairing should resolve");

    assert_eq!(resolved_config.discovered_models.len(), 1);
    assert_eq!(
        resolved_config.discovered_models[0].model_id,
        "Target-Model"
    );
    assert!(
        !resolved_config
            .model_directories
            .contains_key("Target-Model-MTP")
    );
    assert_eq!(resolved_config.mtp_pairings.len(), 1);
    let resolved_pairing = &resolved_config.mtp_pairings[0];
    assert_eq!(resolved_pairing.target_model_id, "Target-Model");
    assert_eq!(resolved_pairing.drafter_model_id, "Target-Model-MTP");
    assert_eq!(
        resolved_pairing.drafter_model_directory.as_ref(),
        Some(&drafter_model_directory)
    );
    assert!(resolved_pairing.discovered_drafter_revision.is_some());
    assert_eq!(
        resolved_config.worker_startup_configuration().mtp_pairings,
        resolved_config.mtp_pairings
    );
}

#[test]
fn should_retain_a_missing_configured_drafter_as_an_unavailable_pairing() {
    let config_home_directory = tempfile::tempdir().expect("config home should be created");
    let model_root_directory = config_home_directory.path().join("models");
    write_shallow_target_model(&model_root_directory.join("Target-Model"));
    write_development_config(
        config_home_directory.path(),
        &format!(
            r#"{{
              "model_directories": [{}],
              "mtp_pairings": [{{
                "target_model_id": "Target-Model",
                "drafter_model_id": "Missing-MTP"
              }}]
            }}"#,
            serde_json::to_string(&model_root_directory).expect("model root should serialize")
        ),
    );
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver
        .load()
        .expect("a missing auxiliary must not remove its target");

    assert_eq!(resolved_config.mtp_pairings.len(), 1);
    assert_eq!(
        resolved_config.mtp_pairings[0].drafter_model_directory,
        None
    );
    assert_eq!(
        resolved_config.mtp_pairings[0].discovered_drafter_revision,
        None
    );
}

#[test]
fn should_reject_a_pairing_whose_target_is_not_publicly_discovered() {
    let config_home_directory = tempfile::tempdir().expect("config home should be created");
    let model_root_directory = config_home_directory.path().join("models");
    write_shallow_mtp_drafter(&model_root_directory.join("Available-MTP"));
    write_development_config(
        config_home_directory.path(),
        &format!(
            r#"{{
              "model_directories": [{}],
              "mtp_pairings": [{{
                "target_model_id": "Missing-Target",
                "drafter_model_id": "Available-MTP"
              }}]
            }}"#,
            serde_json::to_string(&model_root_directory).expect("model root should serialize")
        ),
    );
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    assert!(matches!(
        resolver.load(),
        Err(ResolvedRuntimeConfigError::MtpPairingTargetModelNotDiscovered {
            target_model_id
        }) if target_model_id == "Missing-Target"
    ));
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

    assert!(!resolved_runtime_config.speculative_prefill.is_enabled());
    assert!(
        resolved_runtime_config
            .speculative_prefill_draft_model_directory
            .is_none()
    );
}

#[test]
fn should_reject_enabled_speculative_prefill_when_target_model_is_not_discovered() {
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
                "enabled": true,
                "target_model_id": "target-model",
                "draft_model_id": "draft-model",
                "keep_percentage": 20
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    assert!(matches!(
        resolver.load(),
        Err(ResolvedRuntimeConfigError::SpeculativePrefillTargetModelNotDiscovered { target_model_id })
            if target_model_id == "target-model"
    ));
}

fn write_development_config(home_directory: &std::path::Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical-dev");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config should be written");
}

fn write_shallow_target_model(model_directory: &std::path::Path) {
    std::fs::create_dir_all(model_directory).expect("target directory should be created");
    std::fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5","text_config":{"max_position_embeddings":131072}}"#,
    )
    .expect("target config should be written");
    std::fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("target tokenizer should be written");
    std::fs::write(
        model_directory.join("model.safetensors.index.json"),
        r#"{"weight_map":{}}"#,
    )
    .expect("target index should be written");
}

fn write_shallow_mtp_drafter(model_directory: &std::path::Path) {
    std::fs::create_dir_all(model_directory).expect("drafter directory should be created");
    std::fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_mtp","block_size":4,"text_config":{"hidden_size":1024}}"#,
    )
    .expect("drafter config should be written");
    std::fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("drafter tokenizer should be written");
    std::fs::write(model_directory.join("model.safetensors"), [])
        .expect("drafter payload should be written");
}
