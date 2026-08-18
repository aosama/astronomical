use std::fs;

use astronomical_config::{DiscoveredModelError, discover_models, discover_qwen3_5_mtp_drafters};

fn write_standalone_drafter_config(model_directory: &std::path::Path) {
    fs::write(
        model_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_mtp","block_size":4,"text_config":{"hidden_size":1024}}"#,
    )
    .expect("drafter config should be written");
    fs::write(
        model_directory.join("tokenizer.json"),
        r#"{"version":"1.0"}"#,
    )
    .expect("drafter tokenizer should be written");
}

fn write_single_file_drafter(model_directory: &std::path::Path) {
    write_standalone_drafter_config(model_directory);
    fs::write(model_directory.join("model.safetensors"), [])
        .expect("single-file drafter payload should be written");
}

#[test]
fn should_discover_local_single_file_drafter_without_advertising_it_as_chat_model() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let drafter_directory = temporary_directory.path().join("Qwen-Target-MTP");
    fs::create_dir_all(&drafter_directory).expect("drafter directory should be created");
    write_single_file_drafter(&drafter_directory);
    let configured_roots = vec![temporary_directory.path().to_path_buf()];

    let discovered_drafters = discover_qwen3_5_mtp_drafters(&configured_roots)
        .expect("auxiliary discovery should complete");
    let public_models =
        discover_models(&configured_roots, 20_480).expect("public discovery should complete");

    assert_eq!(discovered_drafters.len(), 1);
    assert_eq!(discovered_drafters[0].model_id, "Qwen-Target-MTP");
    assert_eq!(discovered_drafters[0].model_directory, drafter_directory);
    assert_eq!(discovered_drafters[0].revision.len(), 12);
    assert_eq!(discovered_drafters[0].upstream_revision, None);
    assert!(public_models[0].discovered_models.is_empty());
}

#[test]
fn should_discover_a_complete_indexed_drafter_layout() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let drafter_directory = temporary_directory.path().join("Indexed-MTP");
    fs::create_dir_all(&drafter_directory).expect("drafter directory should be created");
    write_standalone_drafter_config(&drafter_directory);
    fs::write(drafter_directory.join("model-00001.safetensors"), [])
        .expect("drafter shard should be written");
    fs::write(
        drafter_directory.join("model.safetensors.index.json"),
        r#"{"weight_map":{"fc.weight":"model-00001.safetensors"}}"#,
    )
    .expect("drafter index should be written");

    let discovered_drafters =
        discover_qwen3_5_mtp_drafters(&[temporary_directory.path().to_path_buf()])
            .expect("auxiliary discovery should complete");

    assert_eq!(discovered_drafters.len(), 1);
    assert_eq!(discovered_drafters[0].model_id, "Indexed-MTP");
}

#[test]
fn should_discover_the_referenced_hugging_face_snapshot() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let cache_entry_directory = temporary_directory
        .path()
        .join("models--mlx-community--Qwen-Target-MTP");
    let snapshot_directory = cache_entry_directory
        .join("snapshots")
        .join("0123456789abcdef");
    fs::create_dir_all(cache_entry_directory.join("refs"))
        .expect("reference directory should be created");
    fs::create_dir_all(&snapshot_directory).expect("snapshot directory should be created");
    fs::write(
        cache_entry_directory.join("refs/main"),
        "0123456789abcdef\n",
    )
    .expect("main reference should be written");
    write_single_file_drafter(&snapshot_directory);

    let discovered_drafters =
        discover_qwen3_5_mtp_drafters(&[temporary_directory.path().to_path_buf()])
            .expect("auxiliary discovery should complete");

    assert_eq!(discovered_drafters.len(), 1);
    assert_eq!(discovered_drafters[0].model_id, "Qwen-Target-MTP");
    assert_eq!(
        discovered_drafters[0].upstream_revision.as_deref(),
        Some("0123456789abcdef")
    );
    assert_eq!(discovered_drafters[0].revision, "0123456789abcdef");
}

#[test]
fn should_reject_duplicate_auxiliary_model_identities_without_exposing_paths() {
    let first_root = tempfile::tempdir().expect("first root should be created");
    let second_root = tempfile::tempdir().expect("second root should be created");
    for configured_root in [first_root.path(), second_root.path()] {
        let drafter_directory = configured_root.join("Shared-MTP");
        fs::create_dir_all(&drafter_directory).expect("drafter directory should be created");
        write_single_file_drafter(&drafter_directory);
    }

    let discovery_error = discover_qwen3_5_mtp_drafters(&[
        first_root.path().to_path_buf(),
        second_root.path().to_path_buf(),
    ])
    .expect_err("duplicate identity should fail auxiliary discovery");

    assert!(matches!(
        discovery_error,
        DiscoveredModelError::DuplicateAuxiliaryMtpModelId { ref model_id }
            if model_id == "Shared-MTP"
    ));
    assert!(
        !discovery_error
            .to_string()
            .contains(first_root.path().to_string_lossy().as_ref())
    );
    assert!(
        !discovery_error
            .to_string()
            .contains(second_root.path().to_string_lossy().as_ref())
    );
}

#[test]
fn should_skip_incomplete_or_unrelated_auxiliary_artifacts() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let missing_tokenizer_directory = temporary_directory.path().join("Missing-Tokenizer");
    let missing_payload_directory = temporary_directory.path().join("Missing-Payload");
    let unrelated_directory = temporary_directory.path().join("Unrelated");
    for model_directory in [
        &missing_tokenizer_directory,
        &missing_payload_directory,
        &unrelated_directory,
    ] {
        fs::create_dir_all(model_directory).expect("model directory should be created");
    }
    fs::write(
        missing_tokenizer_directory.join("config.json"),
        r#"{"model_type":"qwen3_5_mtp"}"#,
    )
    .expect("config should be written");
    fs::write(missing_tokenizer_directory.join("model.safetensors"), [])
        .expect("payload should be written");
    write_standalone_drafter_config(&missing_payload_directory);
    fs::write(
        unrelated_directory.join("config.json"),
        r#"{"model_type":"qwen3_5"}"#,
    )
    .expect("unrelated config should be written");
    fs::write(unrelated_directory.join("tokenizer.json"), "{}")
        .expect("unrelated tokenizer should be written");
    fs::write(unrelated_directory.join("model.safetensors"), [])
        .expect("unrelated payload should be written");

    let discovered_drafters =
        discover_qwen3_5_mtp_drafters(&[temporary_directory.path().to_path_buf()])
            .expect("auxiliary discovery should complete");

    assert!(discovered_drafters.is_empty());
}
