//! Public Laguna load failures must name the unsupported field or encoding
//! without local paths, so Library and REST can explain why swap failed.

use astronomical_model_serving::{
    LagunaArtifactValidationError, LagunaNormalizationError, LagunaTextArtifactError,
};

#[test]
fn should_name_unsupported_compressed_tensors_storage_in_the_public_load_reason() {
    let validation_error = LagunaArtifactValidationError::Configuration(
        LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: "compressed-tensors".to_owned(),
        },
    );

    assert_eq!(
        validation_error.public_failure_reason(),
        "Laguna artifact uses unsupported storage encoding 'compressed-tensors'"
    );
}

#[test]
fn should_name_the_configuration_field_that_blocked_normalization() {
    let validation_error =
        LagunaArtifactValidationError::Configuration(LagunaNormalizationError::UnsupportedValue {
            field_name: "torch_dtype".to_owned(),
            actual_value: "float64".to_owned(),
        });

    let public_failure_reason = validation_error.public_failure_reason();
    assert!(public_failure_reason.contains("torch_dtype"));
    assert!(public_failure_reason.contains("float64"));
    assert!(!public_failure_reason.contains('/'));
}

#[test]
fn should_name_the_text_sidecar_field_that_blocked_normalization() {
    let validation_error =
        LagunaArtifactValidationError::TextArtifact(LagunaTextArtifactError::InvalidField {
            field_name: "added_tokens_decoder".to_owned(),
        });

    let public_failure_reason = validation_error.public_failure_reason();
    assert!(public_failure_reason.contains("added_tokens_decoder"));
    assert!(!public_failure_reason.contains("/private"));
}

#[test]
fn should_bound_untrusted_public_load_reason_length() {
    let untrusted_encoding = "untrusted-encoding-".repeat(64);
    let validation_error = LagunaArtifactValidationError::Configuration(
        LagunaNormalizationError::UnsupportedStorageEncoding {
            encoding: untrusted_encoding,
        },
    );

    let public_failure_reason = validation_error.public_failure_reason();
    assert!(public_failure_reason.chars().count() <= 512);
    assert!(public_failure_reason.ends_with('…'));
}
