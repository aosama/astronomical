use std::path::PathBuf;

use astronomical_inference_worker::qwen3_5_model_startup_error::Qwen3_5ModelStartupError;
use astronomical_model_serving::{
    ArtifactValidationError, OptiQMetadataError, Qwen3_5ArtifactError,
    Qwen3_5ArtifactValidationError, Qwen3_5ConfigError,
};

#[test]
fn should_explain_the_root_model_validation_failure_without_exposing_its_local_path() {
    let local_model_directory = PathBuf::from("/private/models/example-optiq-model");
    let model_startup_error = Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: local_model_directory.clone(),
        source: Qwen3_5ArtifactValidationError::OptiQMetadata(
            OptiQMetadataError::UnsupportedBits {
                module_name: "language_model.model.layers.5.mlp.switch_mlp.gate_proj".to_owned(),
                actual_bits: 2,
            },
        ),
    };

    let public_failure_reason = model_startup_error.public_model_load_failure_reason();

    assert_eq!(
        public_failure_reason,
        "Qwen3.5 OptiQ metadata validation failed: OptiQ metadata module 'language_model.model.layers.5.mlp.switch_mlp.gate_proj' uses unsupported 2-bit quantization"
    );
    assert!(!public_failure_reason.contains(&local_model_directory.to_string_lossy()[..]));
}

#[test]
fn should_omit_local_model_paths_from_other_artifact_validation_failures() {
    let local_model_directory = PathBuf::from("/private/models/missing-checkpoint");
    let model_startup_error = Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: local_model_directory.clone(),
        source: Qwen3_5ArtifactValidationError::Artifact(
            ArtifactValidationError::ModelDirectoryNotFound {
                model_directory: local_model_directory.clone(),
            },
        ),
    };

    let public_failure_reason = model_startup_error.public_model_load_failure_reason();

    assert_eq!(public_failure_reason, "Qwen3.5 artifact validation failed");
    assert!(!public_failure_reason.contains(&local_model_directory.to_string_lossy()[..]));
}

#[test]
fn should_explain_safe_model_configuration_validation_failures() {
    let model_startup_error = Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: PathBuf::from("/private/models/text-only-qwen"),
        source: Qwen3_5ArtifactValidationError::Config(Qwen3_5ConfigError::MissingActivationDtype),
    };

    assert_eq!(
        model_startup_error.public_model_load_failure_reason(),
        "Qwen3.5 config validation failed: Qwen3.5 config must specify `dtype` at the top level or inside `text_config`"
    );
}

#[test]
fn should_explain_visual_tensors_without_a_vision_config() {
    let model_startup_error = Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: PathBuf::from("/private/models/text-only-qwen"),
        source: Qwen3_5ArtifactValidationError::Qwen3_5ShardIndex(
            Qwen3_5ArtifactError::MissingVisionConfig,
        ),
    };

    assert_eq!(
        model_startup_error.public_model_load_failure_reason(),
        "Qwen3.5 shard-index validation failed: Qwen3.5 index contains visual tensors but config.json has no vision_config"
    );
}

#[test]
fn should_bound_optiq_metadata_failure_reason_length() {
    let untrusted_module_name = "untrusted_module_name_".repeat(64);
    let model_startup_error = Qwen3_5ModelStartupError::ArtifactValidation {
        model_directory: PathBuf::from("/private/models/example-optiq-model"),
        source: Qwen3_5ArtifactValidationError::OptiQMetadata(
            OptiQMetadataError::UnsupportedBits {
                module_name: untrusted_module_name,
                actual_bits: 7,
            },
        ),
    };

    let public_failure_reason = model_startup_error.public_model_load_failure_reason();

    assert!(
        public_failure_reason.chars().count() <= 512,
        "public model-load errors must remain bounded"
    );
    assert!(
        public_failure_reason.ends_with('…'),
        "a truncated public model-load error should identify truncation"
    );
}
