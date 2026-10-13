//! Qwen3.5 artifact validation failures must attribute their cause: the
//! public load-failure reason names the file, tensor, and violated rule
//! instead of a generic wrapper, without leaking local paths, and stays
//! bounded because file names, tensor names, and dtype strings come from
//! untrusted model directories.

use astronomical_model_serving::{ArtifactValidationError, Qwen3_5ArtifactValidationError};
use safetensors::Dtype;

#[test]
fn should_name_the_dtype_rule_a_weight_tensor_violates() {
    let validation_error = ArtifactValidationError::TensorDtypeMismatch {
        tensor_name: "language_model.model.layers.3.linear_attn.out_proj.weight".to_owned(),
        expected_dtype: astronomical_model_serving::TensorDtype::UInt32,
        actual_dtype: Dtype::BF16,
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(
        public_failure_reason.contains("language_model.model.layers.3.linear_attn.out_proj.weight")
    );
    assert!(public_failure_reason.contains("dtype"));
}

#[test]
fn should_name_the_shape_rule_a_weight_tensor_violates() {
    let validation_error = ArtifactValidationError::TensorShapeMismatch {
        tensor_name: "language_model.model.layers.0.self_attn.q_proj.weight".to_owned(),
        expected_shape: vec![2048, 384],
        actual_shape: vec![2048, 512],
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(
        public_failure_reason.contains("language_model.model.layers.0.self_attn.q_proj.weight")
    );
    assert!(public_failure_reason.contains("shape"));
}

#[test]
fn should_name_the_missing_tensor_and_its_weight_file() {
    let validation_error = ArtifactValidationError::TensorMissing {
        tensor_name: "language_model.lm_head.scales".to_owned(),
        file_name: "model-00006-of-00006.safetensors".to_owned(),
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.contains("language_model.lm_head.scales"));
    assert!(public_failure_reason.contains("model-00006-of-00006.safetensors"));
}

#[test]
fn should_name_an_unexpected_tensor_without_a_local_path() {
    let validation_error = ArtifactValidationError::UnexpectedTensor {
        tensor_name: "language_model.model.layers.0.mystery.weight".to_owned(),
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.contains("language_model.model.layers.0.mystery.weight"));
    assert!(!public_failure_reason.contains('/'));
}

#[test]
fn should_name_an_unrecognized_dtype_and_its_tensor() {
    let validation_error = ArtifactValidationError::UnknownSafetensorsDtype {
        file_name: "model-00001-of-00006.safetensors".to_owned(),
        tensor_name: "language_model.model.embed_tokens.weight".to_owned(),
        dtype_string: "F2X".to_owned(),
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.contains("language_model.model.embed_tokens.weight"));
    assert!(public_failure_reason.contains("F2X"));
}

#[test]
fn should_name_a_truncated_weight_file() {
    let validation_error = ArtifactValidationError::TruncatedSafetensorsFile {
        file_name: "model-00003-of-00006.safetensors".to_owned(),
        expected_minimum_bytes: 5_000,
        actual_file_size_bytes: 1_000,
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.contains("model-00003-of-00006.safetensors"));
    assert!(public_failure_reason.contains("truncated"));
}

#[test]
fn should_name_a_required_file_that_does_not_match_its_declared_size() {
    let validation_error = ArtifactValidationError::RequiredFileSizeMismatch {
        file_name: "tokenizer.json".to_owned(),
        expected_size_bytes: 100,
        actual_size_bytes: 40,
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.contains("tokenizer.json"));
    assert!(!public_failure_reason.contains('/'));
}

#[test]
fn should_keep_the_missing_directory_reason_free_of_local_paths() {
    let local_model_directory = std::path::PathBuf::from("/private/models/example-model");
    let validation_error = ArtifactValidationError::ModelDirectoryNotFound {
        model_directory: local_model_directory.clone(),
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(!public_failure_reason.contains(&local_model_directory.to_string_lossy()[..]));
    assert!(
        public_failure_reason.contains("directory"),
        "the reason should still explain that the directory was missing: {public_failure_reason}"
    );
}

#[test]
fn should_bound_the_public_reason_for_untrusted_tensor_names() {
    let untrusted_tensor_name = "untrusted-tensor-".repeat(200);
    let validation_error = ArtifactValidationError::UnexpectedTensor {
        tensor_name: untrusted_tensor_name,
    };

    let public_failure_reason = public_artifact_failure_reason(validation_error);

    assert!(public_failure_reason.chars().count() <= 512);
    assert!(public_failure_reason.ends_with('…'));
}

#[test]
fn should_name_the_safetensors_cause_through_the_qwen_wrapper() {
    let validation_error =
        Qwen3_5ArtifactValidationError::Artifact(ArtifactValidationError::TensorMissing {
            tensor_name: "language_model.lm_head.weight".to_owned(),
            file_name: "model-00006-of-00006.safetensors".to_owned(),
        });

    let public_failure_reason = validation_error.public_failure_reason();

    assert!(public_failure_reason.starts_with("Qwen3.5 artifact validation failed"));
    assert!(public_failure_reason.contains("language_model.lm_head.weight"));
    assert!(!public_failure_reason.contains('/'));
}

#[test]
fn should_preserve_the_existing_config_family_public_reason_through_the_qwen_wrapper() {
    let validation_error = Qwen3_5ArtifactValidationError::Artifact(
        ArtifactValidationError::RequiredFileSizeMismatch {
            file_name: "config.json".to_owned(),
            expected_size_bytes: 100,
            actual_size_bytes: 40,
        },
    );

    assert!(
        validation_error
            .public_failure_reason()
            .contains("config.json")
    );
}

/// Wraps the validation error the way the worker's public load-failure reason
/// composition will, so the tests exercise the composed string rather than a
/// production-only accessor.
fn public_artifact_failure_reason(validation_error: ArtifactValidationError) -> String {
    Qwen3_5ArtifactValidationError::Artifact(validation_error).public_failure_reason()
}
