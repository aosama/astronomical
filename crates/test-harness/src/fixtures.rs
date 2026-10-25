//! Canonical fixture documents shared by the beside-source unit tests and the
//! tests/ integration tree. One copy lives here so the two worlds cannot drift.

use serde_json::json;

// flux2_klein artifact documents

pub fn model_index_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "Flux2KleinPipeline",
        "_diffusers_version": "0.37.0.dev0",
        "is_distilled": true,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"]
    }))
    .expect("the model index should serialize")
}

pub fn transformer_config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "Flux2Transformer2DModel", "_diffusers_version": "0.37.0.dev0",
        "_name_or_path": "/official/build/transformer", "attention_head_dim": 128,
        "axes_dims_rope": [32, 32, 32, 32], "eps": 0.000001, "guidance_embeds": false,
        "in_channels": 128, "joint_attention_dim": 7680, "mlp_ratio": 3.0,
        "num_attention_heads": 24, "num_layers": 5, "num_single_layers": 20,
        "out_channels": null, "patch_size": 1, "rope_theta": 2000,
        "timestep_guidance_channels": 256
    }))
    .expect("the transformer config should serialize")
}

pub fn text_encoder_config_json() -> Vec<u8> {
    let layer_types = vec!["full_attention"; 36];
    serde_json::to_vec(&json!({
        "architectures": ["Qwen3ForCausalLM"], "attention_bias": false,
        "attention_dropout": 0.0, "bos_token_id": 151643, "dtype": "bfloat16",
        "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 2560, "initializer_range": 0.02, "intermediate_size": 9728,
        "layer_types": layer_types, "max_position_embeddings": 40960,
        "max_window_layers": 36, "model_type": "qwen3", "num_attention_heads": 32,
        "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 0.000001,
        "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
        "tie_word_embeddings": true, "transformers_version": "4.56.1",
        "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
    }))
    .expect("the text config should serialize")
}

pub fn vae_config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "AutoencoderKLFlux2", "_diffusers_version": "0.37.0.dev0",
        "_name_or_path": "black-forest-labs/FLUX.2-dev", "act_fn": "silu",
        "batch_norm_eps": 0.0001, "batch_norm_momentum": 0.1,
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": true, "in_channels": 3, "latent_channels": 32,
        "layers_per_block": 2, "mid_block_add_attention": true, "norm_num_groups": 32,
        "out_channels": 3, "patch_size": [2, 2], "sample_size": 1024,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "use_post_quant_conv": true, "use_quant_conv": true
    }))
    .expect("the VAE config should serialize")
}

pub fn scheduler_config_json() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "_class_name": "FlowMatchEulerDiscreteScheduler", "_diffusers_version": "0.37.0.dev0",
        "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": false,
        "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000,
        "shift": 3.0, "shift_terminal": null, "stochastic_sampling": false,
        "time_shift_type": "exponential", "use_beta_sigmas": false,
        "use_dynamic_shifting": true, "use_exponential_sigmas": false,
        "use_karras_sigmas": false
    }))
    .expect("the scheduler config should serialize")
}

pub use e2e_test_model_names::{
    e2e_test_model_ids, flux2_klein_model_id, large_sparse_moe_streaming_model_id,
    ornith_dflash_draft_model_id, required_e2e_test_model_ids, resident_sparse_moe_model_id,
    small_dense_model_id,
};

mod e2e_test_model_names;

pub use qwen_image_21_oracles::{
    QWEN_IMAGE_21_CONDITIONING_ORACLE_JSON, QWEN_IMAGE_21_ROPE_ORACLE_JSON,
    QWEN_IMAGE_21_TIME_ORACLE_JSON,
};

mod qwen_image_21_oracles;
