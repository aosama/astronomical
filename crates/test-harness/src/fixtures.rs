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

// k2_horizon_mova artifact documents
pub fn family_member_config_json(
    hidden_layers: usize,
    mlp_only_layers: &[usize],
    num_experts: usize,
    mova_num_experts: usize,
) -> String {
    serde_json::json!({
        "model_type": "k2_horizon_mova",
        "architectures": ["K2HorizonForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": hidden_layers,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts.min(2),
        "mova_num_experts": mova_num_experts,
        "mova_num_experts_per_tok": mova_num_experts.min(1),
        "num_shared_experts": 1,
        "decoder_sparse_step": 1,
        "mlp_only_layers": mlp_only_layers,
        "rms_norm_eps": 1e-6,
        "layernorm_num_groups": 2,
        "rope_parameters": { "rope_theta": 10_000.0, "rope_type": "default" },
        "attention_bias": false,
        "moe_gate_bias": true,
        "attention_gate_func": "softplus",
        "norm_topk_prob": true,
        "router_score_func": "sigmoid",
        "router_scaling_factor": 2.5,
        "tie_word_embeddings": false,
        "eos_token_id": [1, 2],
        "bos_token_id": 0,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "mode": "affine"
        }
    })
    .to_string()
}
