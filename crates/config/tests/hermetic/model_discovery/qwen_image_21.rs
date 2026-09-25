use std::fs;
use std::path::Path;

use astronomical_config::{
    ModelCapabilities, ModelFamily, QwenImage21DirectoryVerificationError,
    classify_pipeline_index_bytes, verify_qwen_image_21_model_directory,
};
use serde_json::{Value, json};

use super::discover_configured_models;

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
    assert_eq!(
        evidence.license,
        astronomical_config::ModelLicense::QwenResearch
    );
    assert!(evidence.capabilities.supports_text_to_image);
    assert!(!evidence.capabilities.supports_image_editing);
    assert_eq!(evidence.model_size_bytes, fixture_weight_file_size_bytes());
}

#[test]
fn should_advertise_a_verified_qwen_image_2_1_artifact_as_image_generation() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory.path().join("Qwen-Image-Artifact");
    write_qwen_image_21_artifact(&model_directory);

    let discovered_models = &discover_configured_models(&temporary_directory)[0].discovered_models;
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
        qwen_pipeline_index(),
    );
    write_json(
        &model_directory.join("transformer/config.json"),
        qwen_transformer_config(),
    );
    write_json(
        &model_directory.join("text_encoder/config.json"),
        qwen_text_encoder_config(),
    );
    write_json(&model_directory.join("vae/config.json"), qwen_vae_config());
    write_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        qwen_scheduler_config(),
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

fn write_component_weights(component_directory: &Path, payload_size_bytes: usize) {
    write_json(
        &component_directory.join("model.safetensors.index.json"),
        json!({
            "metadata": {"total_size": payload_size_bytes},
            "weight_map": {"model.tensor.0": "model.safetensors"},
        }),
    );
    write_safetensors_payload(
        &component_directory.join("model.safetensors"),
        payload_size_bytes,
    );
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
    qwen_pipeline_index().to_string()
}

fn qwen_pipeline_index() -> Value {
    json!({
        "_class_name": "QwenImage21Pipeline",
        "_diffusers_version": "0.37.0.dev0",
        "processor": ["transformers", "Qwen3VLProcessor"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
    })
}

fn qwen_transformer_config() -> Value {
    json!({
        "_class_name": "QwenImage21Transformer2DModel",
        "attention_head_dim": 128,
        "axes_dims_rope": [16, 56, 56],
        "context_in_dim": 4096,
        "in_channels": 64,
        "num_attention_heads": 32,
        "num_layers": 32,
        "out_channels": 64,
        "patch_size": 1,
        "mlp_ratio": 3,
        "eps": 0.000001,
        "causal_condition": true,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "mlx_format": true,
    })
}

fn qwen_text_encoder_config() -> Value {
    json!({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "qwen3_vl",
        "tie_word_embeddings": false,
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "dtype": "bfloat16",
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.000001,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20],
                "rope_type": "default"
            },
            "rope_theta": 5000000,
            "use_cache": true,
            "vocab_size": 151936
        },
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "mlx_format": true,
    })
}

fn qwen_vae_config() -> Value {
    let channel_values: Vec<f64> = (1_u32..=64).map(|index| f64::from(index)).collect();
    json!({
        "_class_name": "AutoencoderKLQwenImage21",
        "attn_scales": [],
        "base_dim": 96,
        "decoder_base_dim": 144,
        "dim_mult": [1, 2, 4, 8, 8],
        "dropout": 0.0,
        "in_channels": 4,
        "is_residual": true,
        "latents_mean": channel_values,
        "latents_std": channel_values,
        "num_res_blocks": 2,
        "out_channels": 4,
        "patch_size": null,
        "scale_factor_spatial": 16,
        "scale_factor_temporal": 8,
        "temperal_downsample": [false, true, true, true],
        "z_dim": 64,
        "mlx_format": true,
    })
}

fn qwen_scheduler_config() -> Value {
    json!({
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": false,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": false,
        "time_shift_type": "exponential",
        "use_beta_sigmas": false,
        "use_dynamic_shifting": true,
        "use_exponential_sigmas": false,
        "use_karras_sigmas": false,
    })
}

fn write_safetensors_payload(weight_path: &Path, payload_size_bytes: usize) {
    let header_bytes = b"{}";
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + payload_size_bytes, 0);
    fs::write(weight_path, file_bytes).expect("safetensors fixture should be written");
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

fn write_json(file_path: &Path, document: Value) {
    fs::write(
        file_path,
        serde_json::to_vec(&document).expect("fixture JSON should serialize"),
    )
    .expect("fixture JSON should be written");
}
