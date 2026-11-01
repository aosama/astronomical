use std::fs;

use serde_json::{Value, json};

use super::*;
use crate::artifact_validation::validate_required_files;
use crate::qwen3_5_core::frozen_config_test_support::{
    frozen_optiq_metadata_bytes, frozen_ornith_1_0_optiq_config_bytes,
};
use crate::qwen3_5_core::quantizations::producers::optiq::OptiQMetadataError;

fn frozen_config() -> Qwen3_5Config {
    Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("frozen OptiQ config should parse")
}

/// Rewrites the frozen static-shape metadata into the mixed-precision shape.
fn mixed_precision_metadata_bytes() -> Vec<u8> {
    let static_document: Value =
        serde_json::from_slice(&frozen_optiq_metadata_bytes()).expect("frozen metadata decodes");
    serde_json::to_vec(&json!({
        "method": "optiq_mixed_precision",
        "base_model": "synthetic/base",
        "default_bits": 4,
        "group_size": 64,
        "bit_widths_used": [4, 8],
        "layers_per_bit_width": {"4": 1, "8": 1},
        "n_quantized_layers": 510,
        "per_layer": static_document["per_layer"],
    }))
    .expect("mixed metadata serializes")
}

fn validate_directory_with_metadata(
    metadata_relative_path: Option<&str>,
    metadata_bytes: &[u8],
) -> Result<(), Qwen3_5ArtifactValidationError> {
    let model_directory = tempfile::tempdir().expect("temporary model directory");
    if let Some(relative_path) = metadata_relative_path {
        let metadata_path = model_directory.path().join(relative_path);
        fs::create_dir_all(metadata_path.parent().expect("parent")).expect("folder created");
        fs::write(metadata_path, metadata_bytes).expect("metadata written");
    }
    let mut required_file_profiles = Vec::new();
    register_optional_optiq_metadata(&mut required_file_profiles, model_directory.path());
    let required_files = validate_required_files(model_directory.path(), &required_file_profiles)?;
    validate_optional_optiq_metadata(&required_files, &frozen_config())
}

#[test]
fn should_cross_check_metadata_at_the_canonical_location() {
    validate_directory_with_metadata(
        Some("optiq/metadata.json"),
        &mixed_precision_metadata_bytes(),
    )
    .expect("canonical mixed-precision metadata matching the config should pass");
}

#[test]
fn should_cross_check_metadata_at_the_legacy_location() {
    validate_directory_with_metadata(Some("optiq_metadata.json"), &frozen_optiq_metadata_bytes())
        .expect("legacy static metadata matching the config should pass");
}

#[test]
fn should_pass_an_artifact_without_optiq_metadata() {
    validate_directory_with_metadata(None, b"")
        .expect("artifacts from other producers carry no OptiQ metadata");
}

#[test]
fn should_reject_canonical_metadata_that_disagrees_with_the_config() {
    let mut metadata_document: Value =
        serde_json::from_slice(&mixed_precision_metadata_bytes()).expect("decodes");
    let Some(per_layer_profiles) = metadata_document["per_layer"].as_object() else {
        panic!("fixture has a per_layer map");
    };
    let Some(first_module_name) = per_layer_profiles.keys().next().cloned() else {
        panic!("fixture has modules");
    };
    metadata_document["per_layer"][&first_module_name]["bits"] = json!(2);
    let mismatched_bytes = serde_json::to_vec(&metadata_document).expect("serializes");

    let validation_error =
        validate_directory_with_metadata(Some("optiq/metadata.json"), &mismatched_bytes)
            .expect_err("a bit mismatch must be reported");

    assert!(matches!(
        validation_error,
        Qwen3_5ArtifactValidationError::OptiQMetadata(OptiQMetadataError::ConfigBitMismatch { .. })
    ));
}

#[test]
fn should_reject_canonical_metadata_with_unknown_fields() {
    let mut metadata_document: Value =
        serde_json::from_slice(&mixed_precision_metadata_bytes()).expect("decodes");
    metadata_document["surprise"] = json!(true);
    let strict_violation_bytes = serde_json::to_vec(&metadata_document).expect("serializes");

    let validation_error =
        validate_directory_with_metadata(Some("optiq/metadata.json"), &strict_violation_bytes)
            .expect_err("unknown fields must be rejected");

    assert!(matches!(
        validation_error,
        Qwen3_5ArtifactValidationError::OptiQMetadata(OptiQMetadataError::DeserializeMetadata(_))
    ));
}
