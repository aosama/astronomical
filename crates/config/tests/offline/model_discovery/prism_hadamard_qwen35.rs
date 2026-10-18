//! Folded ternary pack discovery: classification, completeness refusals, and
//! the text-only capability projection.

use std::fs;
use std::path::Path;

use astronomical_config::ModelFamily;

/// Writes the smallest executable folded pack: config, Hadamard contract,
/// tokenizer, and single-file weights.
fn write_folded_pack(model_directory: &Path) {
    fs::create_dir_all(model_directory).expect("the pack directory should be created");
    let config_json = serde_json::json!({
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "hadamard_config": "hadamard.json",
        "components": {"text": true, "vision": true, "mtp": false},
        "text_config": {
            "model_type": "qwen3_5_text",
            "max_position_embeddings": 262_144,
        },
    });
    fs::write(model_directory.join("config.json"), config_json.to_string())
        .expect("the pack config should be written");
    fs::write(model_directory.join("hadamard.json"), "{}")
        .expect("the Hadamard contract stub should be written");
    fs::write(model_directory.join("tokenizer.json"), "{}")
        .expect("the tokenizer stub should be written");
    fs::write(model_directory.join("model.safetensors"), [0_u8; 16])
        .expect("the weights stub should be written");
}

#[test]
fn should_classify_a_complete_folded_pack() {
    let configured_root = tempfile::tempdir().expect("the test should create a root");
    let snapshot = configured_root
        .path()
        .join("models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit")
        .join("snapshots")
        .join("fcba37d2117a7077eac6b613b2668d14d9779edd");
    write_folded_pack(&snapshot);

    let artifacts = astronomical_config::discover_classified_model_artifacts(&[configured_root
        .path()
        .to_path_buf()])
    .expect("classified artifact discovery should complete");

    assert_eq!(artifacts.len(), 1);
    assert_eq!(artifacts[0].model_family, ModelFamily::PrismHadamardQwen35);
    assert_eq!(
        artifacts[0].model_id,
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
    );
    // The snapshot directory name is the immutable provenance revision.
    assert_eq!(
        artifacts[0].upstream_revision.as_deref(),
        Some("fcba37d2117a7077eac6b613b2668d14d9779edd")
    );
}

#[test]
fn should_skip_a_folded_pack_missing_its_hadamard_contract() {
    let configured_root = tempfile::tempdir().expect("the test should create a root");
    let snapshot = configured_root.path().join("incomplete-folded-pack");
    write_folded_pack(&snapshot);
    fs::remove_file(snapshot.join("hadamard.json"))
        .expect("the Hadamard contract stub should be removable");

    let directory_scans =
        astronomical_config::discover_models(&[configured_root.path().to_path_buf()])
            .expect("executable model discovery should complete");

    assert!(
        directory_scans
            .iter()
            .all(|scan| scan.discovered_models.is_empty()),
        "a pack without its Hadamard contract must not be advertised as executable"
    );
}

#[test]
fn should_refuse_an_unknown_model_type() {
    let configured_root = tempfile::tempdir().expect("the test should create a root");
    let unrelated_model = configured_root.path().join("unrelated-model");
    fs::create_dir_all(&unrelated_model).expect("the model directory should be created");
    fs::write(
        unrelated_model.join("config.json"),
        r#"{"model_type":"someone_elses_architecture"}"#,
    )
    .expect("the unrelated config should be written");

    let artifacts = astronomical_config::discover_classified_model_artifacts(&[configured_root
        .path()
        .to_path_buf()])
    .expect("classified artifact discovery should complete");

    assert!(artifacts.is_empty());
}
