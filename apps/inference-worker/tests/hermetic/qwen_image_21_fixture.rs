//! Complete shallow Qwen-Image-2.1 fixture used to prove the worker's selected-directory trust
//! boundary. The documents mirror what `verify_qwen_image_21_model_directory` accepts, with
//! placeholder weights whose bytes never matter to the factory's CPU-side checks.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

pub(super) const CANONICAL_MODEL_ID: &str = "Qwen-Image-2.1-MLX-4bit";
pub(super) const REVIEWED_REVISION: &str = "4db4e8c0c0e7a1debf0320415bec8388e888494c";

pub(super) fn write_executable_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "processor",
        "scheduler",
        "text_encoder",
        "transformer",
        "vae",
    ] {
        fs::create_dir_all(model_directory.join(relative_directory))
            .expect("Qwen fixture directory should be created");
    }
    write_json(&model_directory.join("model_index.json"), pipeline_index());
    write_json(
        &model_directory.join("transformer/config.json"),
        transformer_config(),
    );
    write_json(
        &model_directory.join("text_encoder/config.json"),
        text_encoder_config(),
    );
    write_json(&model_directory.join("vae/config.json"), vae_config());
    write_json(
        &model_directory.join("scheduler/scheduler_config.json"),
        scheduler_config(),
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
    write_component_weights(&model_directory.join("text_encoder"), 11);
    write_component_weights(&model_directory.join("transformer"), 17);
    write_component_weights(&model_directory.join("vae"), 23);
    fs::write(
        model_directory.join("README.md"),
        "---\nlicense: other\nlicense_name: qwen-research\nbase_model: Qwen/Qwen-Image-2.1\n---\n# Qwen-Image-2.1\n",
    )
    .expect("Qwen license provenance should be written");
    write_revision(model_directory, REVIEWED_REVISION);
}

pub(super) fn write_revision(model_directory: &Path, revision: &str) {
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{revision}\nfixture-etag\n0\n"),
    )
    .expect("revision metadata should be written");
}

/// Mutates one transformer field so exact-directory verification no longer accepts the artifact
/// while the revision provenance stays intact.
pub(super) fn mutate_profile(model_directory: &Path) {
    let config_path = model_directory.join("transformer/config.json");
    let config_bytes = fs::read(&config_path).expect("transformer config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("transformer config should parse");
    config_document["num_layers"] = json!(31);
    write_json(&config_path, config_document);
}

/// Removes the VAE weights so the component inventory no longer matches the reviewed profile.
pub(super) fn remove_component(model_directory: &Path) {
    fs::remove_file(model_directory.join("vae/model.safetensors"))
        .expect("the VAE weights should be removed");
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

fn write_safetensors_payload(weight_path: &Path, payload_size_bytes: usize) {
    let header_bytes = b"{}";
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + payload_size_bytes, 0);
    fs::write(weight_path, file_bytes).expect("safetensors fixture should be written");
}

fn pipeline_index() -> Value {
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

fn transformer_config() -> Value {
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

fn text_encoder_config() -> Value {
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

fn vae_config() -> Value {
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

fn scheduler_config() -> Value {
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

fn write_json(file_path: &Path, document: Value) {
    fs::write(
        file_path,
        serde_json::to_vec(&document).expect("fixture JSON should serialize"),
    )
    .expect("fixture JSON should be written");
}
