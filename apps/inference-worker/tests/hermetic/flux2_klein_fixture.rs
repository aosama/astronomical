//! Complete shallow FLUX fixture used to prove the worker's selected-directory trust boundary.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

pub(super) const CANONICAL_MODEL_ID: &str = "FLUX.2-klein-4B";
pub(super) const REVIEWED_REVISION: &str = "e7b7dc27f91deacad38e78976d1f2b499d76a294";

pub(super) fn write_executable_artifact(model_directory: &Path) {
    for relative_directory in [
        ".cache/huggingface/download",
        "scheduler",
        "text_encoder",
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
            "_class_name": "Flux2KleinPipeline", "is_distilled": true,
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
        r#"{"metadata":{"total_size":24},"weight_map":{"first":"model-00001-of-00002.safetensors","second":"model-00002-of-00002.safetensors"}}"#,
    )
    .expect("text encoder index should be written");
    for (relative_path, size_bytes) in [
        ("text_encoder/model-00001-of-00002.safetensors", 11),
        ("text_encoder/model-00002-of-00002.safetensors", 13),
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
    write_revision(model_directory, REVIEWED_REVISION);
}

fn write_safetensors_payload(weight_path: &Path, payload_size_bytes: usize) {
    let header_bytes = b"{}";
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + payload_size_bytes, 0);
    fs::write(weight_path, file_bytes).expect("safetensors fixture should be written");
}

pub(super) fn write_revision(model_directory: &Path, revision: &str) {
    fs::write(
        model_directory.join(".cache/huggingface/download/model_index.json.metadata"),
        format!("{revision}\nfixture-etag\n0\n"),
    )
    .expect("revision metadata should be written");
}

pub(super) fn replace_json_field(config_path: &Path, field_name: &str, replacement: Value) {
    let config_bytes = fs::read(config_path).expect("component config should be readable");
    let mut config_document: Value =
        serde_json::from_slice(&config_bytes).expect("component config should parse");
    config_document[field_name] = replacement;
    write_json(config_path, config_document);
}

fn write_json(file_path: &Path, document: Value) {
    fs::write(
        file_path,
        serde_json::to_vec(&document).expect("fixture JSON should serialize"),
    )
    .expect("fixture JSON should be written");
}
