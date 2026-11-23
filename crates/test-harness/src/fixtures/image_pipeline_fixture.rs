//! Shallow image pipeline fixtures: the FLUX.2 klein and Qwen-Image 2.1
//! configuration documents, and the small writers that lay out fixture
//! artifacts on disk. The model-discovery tests in `config` and the worker's
//! trust-boundary tests both build their artifacts from this one copy, so the
//! two cannot disagree about what a well-formed pipeline looks like.

use std::fs;
use std::path::Path;

use serde_json::{Value, json};

/// Writes the five FLUX.2 klein pipeline documents (model index, transformer,
/// text encoder, VAE and scheduler configs) of a shallow fixture artifact.
pub fn write_flux2_klein_pipeline_configs(model_directory: &Path) {
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
}

pub fn qwen_image_21_pipeline_index() -> Value {
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

pub fn qwen_image_21_transformer_config() -> Value {
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

pub fn qwen_image_21_text_encoder_config() -> Value {
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

pub fn qwen_image_21_vae_config() -> Value {
    let channel_values: Vec<f64> = (1_u32..=64).map(f64::from).collect();
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

pub fn qwen_image_21_scheduler_config() -> Value {
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

pub fn write_component_weights(component_directory: &Path, payload_size_bytes: usize) {
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

pub fn write_safetensors_payload(weight_path: &Path, payload_size_bytes: usize) {
    let header_bytes = b"{}";
    let mut file_bytes = (header_bytes.len() as u64).to_le_bytes().to_vec();
    file_bytes.extend_from_slice(header_bytes);
    file_bytes.resize(file_bytes.len() + payload_size_bytes, 0);
    fs::write(weight_path, file_bytes).expect("safetensors fixture should be written");
}

pub fn write_json(file_path: &Path, document: Value) {
    fs::write(
        file_path,
        serde_json::to_vec(&document).expect("fixture JSON should serialize"),
    )
    .expect("fixture JSON should be written");
}
