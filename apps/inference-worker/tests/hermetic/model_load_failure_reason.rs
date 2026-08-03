use std::path::PathBuf;

use astronomical_inference_worker::worker_startup::WorkerStartupError;
use astronomical_model_serving::{
    ArtifactValidationError, OptiQMetadataError, Qwen3_5MoEArtifactError,
    Qwen3_5MoEArtifactValidationError, Qwen3_5MoEConfigError,
};

#[test]
fn should_explain_the_root_model_validation_failure_without_exposing_its_local_path() {
    let local_model_directory = PathBuf::from("/private/models/example-optiq-model");
    let model_startup_error = WorkerStartupError::Qwen3_5MoEArtifactValidation {
        model_directory: local_model_directory.clone(),
        source: Qwen3_5MoEArtifactValidationError::OptiQMetadata(
            OptiQMetadataError::UnsupportedBits {
                module_name: "language_model.model.layers.5.mlp.switch_mlp.gate_proj".to_owned(),
                actual_bits: 2,
            },
        ),
    };

    let public_failure_reason = model_startup_error.public_model_load_failure_reason();

    assert_eq!(
        public_failure_reason,
        "Qwen3.5-MoE OptiQ metadata validation failed: OptiQ metadata module 'language_model.model.layers.5.mlp.switch_mlp.gate_proj' uses unsupported 2-bit quantization"
    );
    assert!(!public_failure_reason.contains(&local_model_directory.to_string_lossy()[..]));
}

#[test]
fn should_omit_local_model_paths_from_other_artifact_validation_failures() {
    let local_model_directory = PathBuf::from("/private/models/missing-checkpoint");
    let model_startup_error = WorkerStartupError::Qwen3_5MoEArtifactValidation {
        model_directory: local_model_directory.clone(),
        source: Qwen3_5MoEArtifactValidationError::Artifact(
            ArtifactValidationError::ModelDirectoryNotFound {
                model_directory: local_model_directory.clone(),
            },
        ),
    };

    let public_failure_reason = model_startup_error.public_model_load_failure_reason();

    assert_eq!(
        public_failure_reason,
        "Qwen3.5-MoE artifact validation failed"
    );
    assert!(!public_failure_reason.contains(&local_model_directory.to_string_lossy()[..]));
}

#[test]
fn should_explain_safe_model_configuration_validation_failures() {
    let model_startup_error = WorkerStartupError::Qwen3_5MoEArtifactValidation {
        model_directory: PathBuf::from("/private/models/text-only-qwen"),
        source: Qwen3_5MoEArtifactValidationError::Config(
            Qwen3_5MoEConfigError::MissingActivationDtype,
        ),
    };

    assert_eq!(
        model_startup_error.public_model_load_failure_reason(),
        "Qwen3.5-MoE config validation failed: Qwen3.5-MoE config must specify `dtype` at the top level or inside `text_config`"
    );
}

#[test]
fn should_explain_visual_tensors_without_a_vision_config() {
    let model_startup_error = WorkerStartupError::Qwen3_5MoEArtifactValidation {
        model_directory: PathBuf::from("/private/models/text-only-qwen"),
        source: Qwen3_5MoEArtifactValidationError::Qwen3_5MoEShardIndex(
            Qwen3_5MoEArtifactError::MissingVisionConfig,
        ),
    };

    assert_eq!(
        model_startup_error.public_model_load_failure_reason(),
        "Qwen3.5-MoE shard-index validation failed: Qwen3.5-MoE index contains visual tensors but config.json has no vision_config"
    );
}

#[test]
fn should_bound_optiq_metadata_failure_reason_length() {
    let untrusted_module_name = "untrusted_module_name_".repeat(64);
    let model_startup_error = WorkerStartupError::Qwen3_5MoEArtifactValidation {
        model_directory: PathBuf::from("/private/models/example-optiq-model"),
        source: Qwen3_5MoEArtifactValidationError::OptiQMetadata(
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
