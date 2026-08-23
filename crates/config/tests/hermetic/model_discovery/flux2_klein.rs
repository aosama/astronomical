use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use astronomical_config::{
    DiscoveredModelError, Flux2KleinDirectoryVerificationError, ModelCapabilities, ModelFamily,
    ModelLicense, verify_flux2_klein_model_directory,
};
use serde_json::{Value, json};

use super::{discover_configured_models, write_minimal_model_config, write_required_model_files};

const CANONICAL_MODEL_ID: &str = "FLUX.2-klein-4B";
const PROVIDER_MODEL_ID: &str = "black-forest-labs/FLUX.2-klein-4B";
const REVIEWED_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";

#[test]
fn should_discover_the_reviewed_distilled_bf16_flux2_klein_profile_with_typed_capabilities() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let model_directory = temporary_directory
        .path()
        .join("Locally-Renamed-Flux-Artifact");
    write_executable_flux2_klein_artifact(&model_directory);

    let directory_scans = discover_configured_models(&temporary_directory);
    let discovered_model = directory_scans[0]
        .discovered_models
        .first()
        .expect("the reviewed FLUX pipeline should be advertised");

    assert_eq!(directory_scans[0].discovered_models.len(), 1);
    assert_eq!(discovered_model.model_id, CANONICAL_MODEL_ID);
    assert_eq!(
        discovered_model.provider_model_id.as_deref(),
        Some(PROVIDER_MODEL_ID)
    );
    assert_eq!(discovered_model.model_family, ModelFamily::Flux2Klein);
    assert_eq!(discovered_model.revision, REVIEWED_REVISION);
    assert_eq!(discovered_model.license, Some(ModelLicense::Apache20));
    assert!(discovered_model.model_size_bytes > 0);
    assert_eq!(
        discovered_model.model_size_bytes,
        modular_weight_size_from_fixture(&model_directory)
    );
    let ModelCapabilities::ImageGeneration(image_capabilities) = &discovered_model.capabilities
    else {
        panic!("FLUX discovery must expose image-generation capabilities without token limits");
    };
    assert!(image_capabilities.supports_text_to_image);
    assert!(!image_capabilities.supports_image_editing);
    assert!(!image_capabilities.supports_multiple_reference_images);
}

#[test]
fn should_verify_exact_directory_evidence_without_scanning_a_parent_root() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let selected_model_directory = temporary_directory.path().join("selected-model");
    let unrelated_model_directory = temporary_directory.path().join("unrelated-model");
    write_executable_flux2_klein_artifact(&selected_model_directory);
    write_executable_flux2_klein_artifact(&unrelated_model_directory);

    let verified_evidence = verify_flux2_klein_model_directory(&selected_model_directory)
        .expect("the selected complete directory should verify");

    assert_eq!(verified_evidence.canonical_model_id, CANONICAL_MODEL_ID);
    assert_eq!(verified_evidence.provider_model_id, PROVIDER_MODEL_ID);
    assert_eq!(verified_evidence.revision, REVIEWED_REVISION);
    assert_eq!(verified_evidence.license, ModelLicense::Apache20);
    assert!(verified_evidence.capabilities.supports_text_to_image);
    assert!(!verified_evidence.capabilities.supports_image_editing);
    assert!(
        !verified_evidence
            .capabilities
            .supports_multiple_reference_images
    );

    fs::remove_file(selected_model_directory.join("vae/diffusion_pytorch_model.safetensors"))
        .expect("selected component should be removed");
    assert_eq!(
        verify_flux2_klein_model_directory(&selected_model_directory),
        Err(Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile { component: "vae" })
    );
}

#[test]
fn should_report_bounded_path_free_flux_verification_failures() {
    for (invalid_artifact, expected_error) in [
        (
            InvalidArtifact::MalformedPipeline,
            Flux2KleinDirectoryVerificationError::InvalidPipelineIndex,
        ),
        (
            InvalidArtifact::WrongLicense,
            Flux2KleinDirectoryVerificationError::InvalidLicense,
        ),
        (
            InvalidArtifact::WrongRevision,
            Flux2KleinDirectoryVerificationError::UnexpectedRevision,
        ),
        (
            InvalidArtifact::EmptyWeightMap,
            Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex,
        ),
        (
            InvalidArtifact::UnsafeShardPath,
            Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex,
        ),
        (
            InvalidArtifact::MismatchedTextEncoderTotalSize,
            Flux2KleinDirectoryVerificationError::InvalidTextEncoderWeightIndex,
        ),
        (
            InvalidArtifact::MissingTextEncoderShard,
            Flux2KleinDirectoryVerificationError::MissingOrInvalidWeightFile {
                component: "text encoder",
            },
        ),
    ] {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        let model_directory = temporary_directory.path().join("Rejected-Flux-Artifact");
        write_executable_flux2_klein_artifact(&model_directory);
        invalidate_artifact(&model_directory, invalid_artifact);

        let verification_error = verify_flux2_klein_model_directory(&model_directory)
            .expect_err("invalid evidence should retain its typed failure category");
        assert_eq!(verification_error, expected_error);
        assert!(
            !verification_error.to_string().contains(
                model_directory
                    .to_str()
                    .expect("temporary path should be UTF-8")
            )
        );
    }
}

#[test]
fn should_reject_malformed_wrong_profile_license_or_revision_evidence() {
    for invalid_artifact in [
        InvalidArtifact::MalformedPipeline,
        InvalidArtifact::BaseProfile,
        InvalidArtifact::WrongDtype,
        InvalidArtifact::WrongLicense,
        InvalidArtifact::WrongRevision,
        InvalidArtifact::MissingComponent,
        InvalidArtifact::EmptyWeightMap,
        InvalidArtifact::UnsafeShardPath,
        InvalidArtifact::MismatchedTextEncoderTotalSize,
        InvalidArtifact::MissingTextEncoderShard,
    ] {
        let temporary_directory =
            tempfile::tempdir().expect("temporary directory should be created");
        let model_directory = temporary_directory.path().join("Rejected-Flux-Artifact");
        write_executable_flux2_klein_artifact(&model_directory);
        invalidate_artifact(&model_directory, invalid_artifact);

        assert!(
            discover_configured_models(&temporary_directory)[0]
                .discovered_models
                .is_empty(),
            "discovery must reject {invalid_artifact:?}"
        );
    }
}

#[test]
fn should_stop_at_a_pipeline_root_instead_of_discovering_nested_component_configs() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let pipeline_directory = temporary_directory.path().join("Unsupported-Pipeline");
    let nested_component_directory = pipeline_directory.join("text_encoder");
    fs::create_dir_all(&nested_component_directory).expect("component directory should be created");
    fs::write(
        pipeline_directory.join("model_index.json"),
        r#"{"_class_name":"UnsupportedPipeline"}"#,
    )
    .expect("unsupported pipeline index should be written");
    write_minimal_model_config(&nested_component_directory, "qwen3_5", 4_096);
    write_required_model_files(&nested_component_directory);

    assert!(
        discover_configured_models(&temporary_directory)[0]
            .discovered_models
            .is_empty()
    );
}

#[test]
fn should_reject_two_flux_artifacts_that_share_the_canonical_serving_identity() {
    let temporary_directory = tempfile::tempdir().expect("temporary directory should be created");
    let first_root = temporary_directory.path().join("first-root");
    let second_root = temporary_directory.path().join("second-root");
    write_executable_flux2_klein_artifact(&first_root.join("first-local-name"));
    write_executable_flux2_klein_artifact(&second_root.join("second-local-name"));

    let discovery_error = astronomical_config::discover_models(&[first_root, second_root])
        .expect_err("duplicate canonical FLUX identities should be rejected");

    assert!(matches!(
        discovery_error,
        DiscoveredModelError::DuplicateModelId { model_id, .. } if model_id == CANONICAL_MODEL_ID
    ));
}

#[derive(Clone, Copy, Debug)]
enum InvalidArtifact {
    MalformedPipeline,
    BaseProfile,
    WrongDtype,
    WrongLicense,
    WrongRevision,
    MissingComponent,
    EmptyWeightMap,
    UnsafeShardPath,
    MismatchedTextEncoderTotalSize,
    MissingTextEncoderShard,
}

fn invalidate_artifact(model_directory: &Path, invalid_artifact: InvalidArtifact) {
    match invalid_artifact {
        InvalidArtifact::MalformedPipeline => fs::write(
            model_directory.join("model_index.json"),
            r#"{"_class_name":"Flux2KleinPipeline"#,
        )
        .expect("malformed pipeline should be written"),
        InvalidArtifact::BaseProfile => replace_json_field(
            &model_directory.join("model_index.json"),
            "is_distilled",
            json!(false),
        ),
        InvalidArtifact::WrongDtype => replace_json_field(
            &model_directory.join("text_encoder/config.json"),
            "dtype",
            json!("float16"),
        ),
        InvalidArtifact::WrongLicense => {
            fs::write(
                model_directory.join("LICENSE.md"),
                "Fictional model license",
            )
            .expect("wrong license should be written");
        }
        InvalidArtifact::WrongRevision => fs::write(
            model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
            "0123456789abcdef0123456789abcdef01234567\nfixture-etag\n0\n",
        )
        .expect("wrong immutable revision should be written"),
        InvalidArtifact::MissingComponent => {
            fs::remove_file(model_directory.join("transformer/diffusion_pytorch_model.safetensors"))
                .expect("required component should be removed")
        }
        InvalidArtifact::EmptyWeightMap => replace_json_field(
            &model_directory.join("text_encoder/model.safetensors.index.json"),
            "weight_map",
            json!({}),
        ),
        InvalidArtifact::UnsafeShardPath => replace_json_field(
            &model_directory.join("text_encoder/model.safetensors.index.json"),
            "weight_map",
            json!({"tensor": "../outside.safetensors"}),
        ),
        InvalidArtifact::MismatchedTextEncoderTotalSize => replace_index_total_size(
            &model_directory.join("text_encoder/model.safetensors.index.json"),
            23,
        ),
        InvalidArtifact::MissingTextEncoderShard => {
            fs::remove_file(model_directory.join("text_encoder/weights/encoder-part-b.safetensors"))
                .expect("indexed text encoder shard should be removed")
        }
    }
}

fn write_executable_flux2_klein_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "scheduler",
        "text_encoder",
        "text_encoder/weights",
        "tokenizer",
        "transformer",
        "vae",
    ] {
        fs::create_dir_all(model_directory.join(relative_directory))
            .expect("FLUX fixture directory should be created");
    }
    write_json(
        &model_directory.join("model_index.json"),
        json!({
            "_class_name": "Flux2KleinPipeline",
            "is_distilled": true,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }),
    );
    write_json(
        &model_directory.join("transformer/config.json"),
        json!({
            "_class_name": "Flux2Transformer2DModel", "attention_head_dim": 128,
            "axes_dims_rope": [32, 32, 32, 32], "eps": 0.000001,
            "guidance_embeds": false, "in_channels": 128, "joint_attention_dim": 7680,
            "mlp_ratio": 3.0, "num_attention_heads": 24, "num_layers": 5,
            "num_single_layers": 20, "out_channels": null, "patch_size": 1,
            "rope_theta": 2000, "timestep_guidance_channels": 256,
        }),
    );
    write_json(
        &model_directory.join("text_encoder/config.json"),
        json!({
            "architectures": ["Qwen3ForCausalLM"], "attention_bias": false,
            "attention_dropout": 0.0, "dtype": "bfloat16", "head_dim": 128,
            "hidden_act": "silu", "hidden_size": 2560, "intermediate_size": 9728,
            "layer_types": vec!["full_attention"; 36], "max_position_embeddings": 40960,
            "max_window_layers": 36, "model_type": "qwen3", "num_attention_heads": 32,
            "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 0.000001,
            "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
            "tie_word_embeddings": true, "use_cache": true,
            "use_sliding_window": false, "vocab_size": 151936,
        }),
    );
    write_json(
        &model_directory.join("vae/config.json"),
        json!({
            "_class_name": "AutoencoderKLFlux2", "act_fn": "silu",
            "batch_norm_eps": 0.0001, "batch_norm_momentum": 0.1,
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            "force_upcast": true, "in_channels": 3, "latent_channels": 32,
            "layers_per_block": 2, "mid_block_add_attention": true,
            "norm_num_groups": 32, "out_channels": 3, "patch_size": [2, 2],
            "sample_size": 1024,
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            "use_post_quant_conv": true, "use_quant_conv": true,
        }),
    );
    write_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        json!({
            "_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256,
            "base_shift": 0.5, "invert_sigmas": false, "max_image_seq_len": 4096,
            "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0,
            "shift_terminal": null, "stochastic_sampling": false,
            "time_shift_type": "exponential", "use_beta_sigmas": false,
            "use_dynamic_shifting": true, "use_exponential_sigmas": false,
            "use_karras_sigmas": false,
        }),
    );
    fs::write(
        model_directory.join("text_encoder/model.safetensors.index.json"),
        r#"{"metadata":{"total_size":24},"weight_map":{"first":"weights/encoder-part-a.safetensors","second":"weights/encoder-part-b.safetensors","shared":"weights/encoder-part-a.safetensors"}}"#,
    )
    .expect("text encoder index should be written");
    for (relative_path, size_bytes) in [
        ("text_encoder/weights/encoder-part-a.safetensors", 11),
        ("text_encoder/weights/encoder-part-b.safetensors", 13),
    ] {
        write_safetensors_payload(&model_directory.join(relative_path), size_bytes);
    }
    for (relative_path, size_bytes) in [
        ("transformer/diffusion_pytorch_model.safetensors", 17),
        ("vae/diffusion_pytorch_model.safetensors", 13),
    ] {
        fs::write(model_directory.join(relative_path), vec![0_u8; size_bytes])
            .expect("modular weight should be written");
    }
    for relative_path in [
        "text_encoder/generation_config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
    ] {
        fs::write(model_directory.join(relative_path), "{}")
            .expect("modular sidecar should be written");
    }
    fs::write(
        model_directory.join("LICENSE.md"),
        "Apache License\nVersion 2.0, January 2004\nTERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\nEND OF TERMS AND CONDITIONS\n",
    )
    .expect("Apache-2.0 license should be written");
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{REVIEWED_REVISION}\nfixture-etag\n0\n"),
    )
    .expect("reviewed revision metadata should be written");
}

fn modular_weight_size_from_fixture(model_directory: &Path) -> u64 {
    let index_bytes = fs::read(model_directory.join("text_encoder/model.safetensors.index.json"))
        .expect("text encoder index should be readable");
    let index_document: Value =
        serde_json::from_slice(&index_bytes).expect("text encoder index should parse");
    let unique_shard_paths = index_document["weight_map"]
        .as_object()
        .expect("weight map should be an object")
        .values()
        .map(|shard_path| {
            shard_path
                .as_str()
                .expect("fixture shard path should be a string")
        })
        .collect::<BTreeSet<_>>();
    assert!(!unique_shard_paths.is_empty());
    let text_encoder_size_bytes = unique_shard_paths
        .into_iter()
        .map(|shard_path| {
            fs::metadata(model_directory.join("text_encoder").join(shard_path))
                .expect("indexed shard should have metadata")
                .len()
        })
        .sum::<u64>();
    assert!(
        index_document["metadata"]["total_size"]
            .as_u64()
            .expect("fixture total size should be an unsigned integer")
            < text_encoder_size_bytes
    );
    text_encoder_size_bytes
        + fs::metadata(model_directory.join("transformer/diffusion_pytorch_model.safetensors"))
            .expect("transformer weight should have metadata")
            .len()
        + fs::metadata(model_directory.join("vae/diffusion_pytorch_model.safetensors"))
            .expect("VAE weight should have metadata")
            .len()
}

fn write_safetensors_payload(weight_path: &Path, payload_size_bytes: usize) {
    let header_bytes = b"{}";
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + payload_size_bytes, 0);
    fs::write(weight_path, file_bytes).expect("safetensors fixture should be written");
}

fn replace_index_total_size(index_path: &Path, invalid_total_size: u64) {
    let index_bytes = fs::read(index_path).expect("text encoder index should be readable");
    let mut index_document: Value =
        serde_json::from_slice(&index_bytes).expect("text encoder index should parse");
    index_document["metadata"]["total_size"] = json!(invalid_total_size);
    write_json(index_path, index_document);
}

fn replace_json_field(config_path: &Path, field_name: &str, invalid_value: Value) {
    let config_bytes = fs::read(config_path).expect("component config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("component config should parse");
    config_document[field_name] = invalid_value;
    write_json(config_path, config_document);
}

fn write_json(file_path: &Path, document: Value) {
    fs::write(
        file_path,
        serde_json::to_vec(&document).expect("fixture JSON should serialize"),
    )
    .expect("fixture JSON should be written");
}
