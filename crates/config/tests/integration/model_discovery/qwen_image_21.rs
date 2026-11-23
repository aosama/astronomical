use std::fs;
use std::path::Path;

use config::{
    ModelCapabilities, ModelFamily, QwenImage21DirectoryVerificationError,
    classify_pipeline_index_bytes, verify_qwen_image_21_model_directory,
};
use serde_json::{Value, json};
use test_harness::fixtures::image_pipeline_fixture::{
    qwen_image_21_pipeline_index, qwen_image_21_scheduler_config,
    qwen_image_21_text_encoder_config, qwen_image_21_transformer_config, qwen_image_21_vae_config,
    write_component_weights, write_json,
};

const CANONICAL_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
const PROVIDER_MODEL_ID: &str = "mlx-community/Qwen-Image-2.1-MLX-4bit";
const REVIEWED_REVISION: &str = "4db4e8c0c0e7a1debf0320415bec8388e888494c";
const TEXT_ENCODER_PAYLOAD_BYTES: usize = 11;
const TRANSFORMER_PAYLOAD_BYTES: usize = 17;
const VAE_PAYLOAD_BYTES: usize = 23;

#[test]
fn should_classify_the_reviewed_qwen_image_21_pipeline() {
    let pipeline_index = qwen_pipeline_index_json();

    assert_eq!(
        classify_pipeline_index_bytes(pipeline_index.as_bytes())
            .expect("fixture pipeline should parse"),
        Some(ModelFamily::QwenImage21)
    );
}

#[test]
fn should_verify_complete_qwen_image_21_profile_evidence() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Local-Qwen-Image-Artifact");
    write_qwen_image_21_artifact(&model_directory);

    let evidence = verify_qwen_image_21_model_directory(&model_directory)
        .expect("complete reviewed profile should verify");

    assert_eq!(evidence.canonical_model_id, CANONICAL_MODEL_ID);
    assert_eq!(evidence.provider_model_id, PROVIDER_MODEL_ID);
    assert_eq!(evidence.revision, REVIEWED_REVISION);
    assert_eq!(evidence.license, config::ModelLicense::QwenResearch);
    assert!(evidence.capabilities.supports_text_to_image);
    assert!(!evidence.capabilities.supports_image_editing);
    assert_eq!(evidence.capabilities.default_steps, 40);
    assert_eq!(evidence.capabilities.minimum_dimension_pixels, 256);
    assert_eq!(evidence.capabilities.maximum_dimension_pixels, 1_024);
    assert_eq!(evidence.capabilities.dimension_multiple_pixels, 32);
    assert_eq!(evidence.model_size_bytes, fixture_weight_file_size_bytes());
}

#[test]
fn should_advertise_a_verified_qwen_image_2_1_artifact_as_image_generation() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Qwen-Image-Artifact");
    write_qwen_image_21_artifact(&model_directory);

    let discovered_models =
        &super::discover_configured_models(&temporary_directory)[0].discovered_models;
    let discovered_model = discovered_models
        .iter()
        .find(|discovered_model| discovered_model.model_family == ModelFamily::QwenImage21)
        .expect("a verified Qwen-Image-2.1 artifact must appear in the executable model list");

    assert_eq!(discovered_model.model_id, CANONICAL_MODEL_ID);
    assert_eq!(
        discovered_model.provider_model_id.as_deref(),
        Some(PROVIDER_MODEL_ID)
    );
    assert_eq!(discovered_model.revision, REVIEWED_REVISION);
    assert!(
        matches!(
            &discovered_model.capabilities,
            ModelCapabilities::ImageGeneration(capabilities)
                if capabilities.supports_text_to_image
        ),
        "the native text-to-image engine is executable, so discovery advertises the capability"
    );
}

#[test]
fn should_report_bounded_path_free_qwen_profile_failures() {
    let invalid_cases = [
        (
            InvalidQwenArtifact::WrongPipelineClass,
            QwenImage21DirectoryVerificationError::InvalidPipelineIndex,
        ),
        (
            InvalidQwenArtifact::WrongScheduler,
            QwenImage21DirectoryVerificationError::InvalidSchedulerConfiguration,
        ),
        (
            InvalidQwenArtifact::MissingProcessorFile,
            QwenImage21DirectoryVerificationError::MissingOrInvalidProcessorFile {
                processor_file: "processor/chat_template.jinja",
            },
        ),
        (
            InvalidQwenArtifact::WrongLicenseProvenance,
            QwenImage21DirectoryVerificationError::InvalidLicenseProvenance,
        ),
        (
            InvalidQwenArtifact::UnsafeShardPath,
            QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: "text encoder",
            },
        ),
        (
            InvalidQwenArtifact::MismatchedWeightIndex,
            QwenImage21DirectoryVerificationError::InvalidComponentWeightIndex {
                component: "transformer",
            },
        ),
        (
            InvalidQwenArtifact::MissingWeight,
            QwenImage21DirectoryVerificationError::MissingOrInvalidWeightFile { component: "VAE" },
        ),
    ];
    for (invalid_artifact, expected_error) in invalid_cases {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        let model_directory = temporary_directory.path().join("Rejected-Qwen-Artifact");
        write_qwen_image_21_artifact(&model_directory);
        invalidate_artifact(&model_directory, invalid_artifact);

        let verification_error = verify_qwen_image_21_model_directory(&model_directory)
            .expect_err("invalid profile evidence should retain its typed failure");
        assert_eq!(verification_error, expected_error);
        assert!(
            !verification_error
                .to_string()
                .contains(model_directory.to_string_lossy().as_ref()),
            "verification errors must not expose local paths"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum InvalidQwenArtifact {
    WrongPipelineClass,
    WrongScheduler,
    MissingProcessorFile,
    WrongLicenseProvenance,
    UnsafeShardPath,
    MismatchedWeightIndex,
    MissingWeight,
}

fn invalidate_artifact(model_directory: &Path, invalid_artifact: InvalidQwenArtifact) {
    match invalid_artifact {
        InvalidQwenArtifact::WrongPipelineClass => replace_json_field(
            &model_directory.join("model_index.json"),
            "_class_name",
            json!("QwenImagePipeline"),
        ),
        InvalidQwenArtifact::WrongScheduler => replace_json_field(
            &model_directory.join("scheduler/scheduler_config.json"),
            "max_image_seq_len",
            json!(4096),
        ),
        InvalidQwenArtifact::MissingProcessorFile => {
            fs::remove_file(model_directory.join("processor/chat_template.jinja"))
                .expect("processor fixture should be removable")
        }
        InvalidQwenArtifact::WrongLicenseProvenance => fs::write(
            model_directory.join("README.md"),
            "---\nlicense: apache-2.0\n---\n",
        )
        .expect("wrong license provenance should be written"),
        InvalidQwenArtifact::UnsafeShardPath => replace_json_field(
            &model_directory.join("text_encoder/model.safetensors.index.json"),
            "weight_map",
            json!({"tensor": "../outside.safetensors"}),
        ),
        InvalidQwenArtifact::MismatchedWeightIndex => write_json(
            &model_directory.join("transformer/model.safetensors.index.json"),
            json!({
                "metadata": {"total_size": 1},
                "weight_map": {"model.tensor.0": "model.safetensors"},
            }),
        ),
        InvalidQwenArtifact::MissingWeight => {
            fs::remove_file(model_directory.join("vae/model.safetensors"))
                .expect("VAE weight fixture should be removable");
        }
    }
}

fn write_qwen_image_21_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "processor",
        "scheduler",
        "text_encoder",
        "transformer",
        "vae",
    ] {
        fs::create_dir_all(model_directory.join(relative_directory))
            .expect("Qwen artifact directory should be created");
    }
    write_json(
        &model_directory.join("model_index.json"),
        qwen_image_21_pipeline_index(),
    );
    write_json(
        &model_directory.join("transformer/config.json"),
        qwen_image_21_transformer_config(),
    );
    write_json(
        &model_directory.join("text_encoder/config.json"),
        qwen_image_21_text_encoder_config(),
    );
    write_json(
        &model_directory.join("vae/config.json"),
        qwen_image_21_vae_config(),
    );
    write_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        qwen_image_21_scheduler_config(),
    );
    for processor_file in [
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ] {
        fs::write(
            model_directory.join("processor").join(processor_file),
            "fixture",
        )
        .expect("processor fixture should be written");
    }
    write_component_weights(
        &model_directory.join("text_encoder"),
        TEXT_ENCODER_PAYLOAD_BYTES,
    );
    write_component_weights(
        &model_directory.join("transformer"),
        TRANSFORMER_PAYLOAD_BYTES,
    );
    write_component_weights(&model_directory.join("vae"), VAE_PAYLOAD_BYTES);
    fs::write(
        model_directory.join("README.md"),
        "---\nlicense: other\nlicense_name: qwen-research\nbase_model: Qwen/Qwen-Image-2.1\n---\n# Qwen-Image-2.1\n",
    )
    .expect("Qwen license provenance should be written");
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{REVIEWED_REVISION}\nfixture-etag\n0\n"),
    )
    .expect("immutable revision should be written");
}

fn fixture_weight_file_size_bytes() -> u64 {
    [
        TEXT_ENCODER_PAYLOAD_BYTES,
        TRANSFORMER_PAYLOAD_BYTES,
        VAE_PAYLOAD_BYTES,
    ]
    .into_iter()
    .map(|payload_size_bytes| (payload_size_bytes + 10) as u64)
    .sum()
}

fn qwen_pipeline_index_json() -> String {
    qwen_image_21_pipeline_index().to_string()
}

fn replace_json_field(document_path: &Path, field_name: &str, invalid_value: Value) {
    let document_bytes = fs::read(document_path).expect("fixture document should be readable");
    let mut document: Value =
        serde_json::from_slice(&document_bytes).expect("fixture document should parse");
    document
        .as_object_mut()
        .expect("fixture document should be an object")
        .insert(field_name.to_owned(), invalid_value);
    write_json(document_path, document);
}
