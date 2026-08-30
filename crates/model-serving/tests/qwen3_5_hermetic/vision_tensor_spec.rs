use astronomical_model_serving::{Qwen3_5VisionConfig, qwen3_5_vision_tensor_profiles};

const FROZEN_VISION_CONFIG_JSON: &str = r#"{
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "dtype": "bfloat16",
    "eos_token_id": [248046, 248044],
    "tie_word_embeddings": false,
    "quantization": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "deepreinforce-ai/Ornith-1.0-35B"},
    "quantization_config": {"config_file": "optiq/optiq.safetensors", "n_tensors": 333, "base_model": "deepreinforce-ai/Ornith-1.0-35B"},
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_act": "silu",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
        "attention_bias": false,
        "mlp_bias": false,
        "norm_topk_prob": true,
        "output_router_logits": false,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "mtp_num_hidden_layers": 1
    },
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }
}"#;

#[test]
fn should_generate_the_complete_ornith_vision_tensor_profile() {
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(FROZEN_VISION_CONFIG_JSON.as_bytes())
        .expect("the frozen Ornith 1.0 vision config should parse");
    let vision_tensor_profiles = qwen3_5_vision_tensor_profiles(&vision_config);

    // 333 total tensors: 27 blocks x 12 + 3 patch/pos + 6 merger
    assert_eq!(
        vision_tensor_profiles.len(),
        333,
        "vision tensor profile must have exactly 333 tensors"
    );

    // Patch embed
    let patch_weight = &vision_tensor_profiles[0];
    assert_eq!(patch_weight.name, "vision_tower.patch_embed.proj.weight");
    assert_eq!(patch_weight.shape, vec![1152, 2, 16, 16, 3]);

    let patch_bias = &vision_tensor_profiles[1];
    assert_eq!(patch_bias.name, "vision_tower.patch_embed.proj.bias");
    assert_eq!(patch_bias.shape, vec![1152]);

    // Positional embedding
    let pos_embed = &vision_tensor_profiles[2];
    assert_eq!(pos_embed.name, "vision_tower.pos_embed.weight");
    assert_eq!(pos_embed.shape, vec![2304, 1152]);

    // First block attention
    let block0_qkv_weight = &vision_tensor_profiles[3];
    assert_eq!(
        block0_qkv_weight.name,
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(block0_qkv_weight.shape, vec![3456, 1152]);

    // Last block norm2 bias (block 26, 12th tensor = index 3 + 26*12 + 11)
    let block26_norm2_bias = &vision_tensor_profiles[3 + 26 * 12 + 11];
    assert_eq!(block26_norm2_bias.name, "vision_tower.blocks.26.norm2.bias");
    assert_eq!(block26_norm2_bias.shape, vec![1152]);

    // Merger (last 6 tensors: indices 327..333)
    let merger_norm_weight = &vision_tensor_profiles[327];
    assert_eq!(merger_norm_weight.name, "vision_tower.merger.norm.weight");
    assert_eq!(merger_norm_weight.shape, vec![1152]);

    let merger_fc1_weight = &vision_tensor_profiles[329];
    assert_eq!(
        merger_fc1_weight.name,
        "vision_tower.merger.linear_fc1.weight"
    );
    assert_eq!(merger_fc1_weight.shape, vec![4608, 4608]);

    let merger_fc2_weight = &vision_tensor_profiles[331];
    assert_eq!(
        merger_fc2_weight.name,
        "vision_tower.merger.linear_fc2.weight"
    );
    assert_eq!(merger_fc2_weight.shape, vec![2048, 4608]);

    let merger_fc2_bias = &vision_tensor_profiles[332];
    assert_eq!(merger_fc2_bias.name, "vision_tower.merger.linear_fc2.bias");
    assert_eq!(merger_fc2_bias.shape, vec![2048]);

    // Every stored floating tensor must retain an MLX-supported model dtype.
    for tensor_profile in &vision_tensor_profiles {
        assert_eq!(
            tensor_profile.dtype,
            astronomical_model_serving::TensorDtype::ModelFloat,
            "tensor {} must retain a supported stored model float dtype",
            tensor_profile.name
        );
    }
}
