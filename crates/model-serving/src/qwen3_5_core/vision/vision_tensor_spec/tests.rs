use super::*;
use crate::TensorDtype;
use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes;
use serde_json::{Value, json};

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
        "shared_expert_intermediate_size": 512
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
    let vision_tensor_profiles = super::qwen3_5_vision_tensor_profiles(&vision_config);

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
    assert_eq!(
        patch_weight.equivalent_published_shapes,
        vec![vec![1152, 3, 2, 16, 16]],
        "the upstream PyTorch Conv3d order must be accepted and normalized to ODHWI"
    );

    let patch_bias = &vision_tensor_profiles[1];
    assert_eq!(patch_bias.name, "vision_tower.patch_embed.proj.bias");
    assert_eq!(patch_bias.shape, vec![1152]);
    assert!(patch_bias.equivalent_published_shapes.is_empty());

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
            TensorDtype::ModelFloat,
            "tensor {} must retain a supported stored model float dtype",
            tensor_profile.name
        );
    }
}

#[test]
fn should_generate_mlx_affine_vision_companions_from_module_profiles() {
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(FROZEN_VISION_CONFIG_JSON.as_bytes())
        .expect("the frozen Ornith 1.0 vision config should parse");
    let qwen3_5_config = qwen_config_with_native_vision_mlp_output();

    let vision_tensor_profiles =
        qwen3_5_vision_tensor_profiles_for_model(&vision_config, &qwen3_5_config)
            .expect("affine vision profiles should derive from the configured weight contract");
    let tensor_profile_by_name = vision_tensor_profiles
        .iter()
        .map(|tensor_profile| (tensor_profile.name.as_str(), tensor_profile))
        .collect::<std::collections::HashMap<_, _>>();

    assert_eq!(vision_tensor_profiles.len(), 499);
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.attn.qkv.weight"].dtype,
        TensorDtype::UInt32
    );
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.attn.qkv.weight"].shape,
        [3456, 216]
    );
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.attn.qkv.scales"].shape,
        [3456, 18]
    );
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.attn.qkv.biases"].shape,
        [3456, 18]
    );
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.attn.qkv.bias"].shape,
        [3456]
    );
    assert_eq!(
        tensor_profile_by_name["vision_tower.blocks.0.mlp.linear_fc2.weight"].dtype,
        TensorDtype::ModelFloat
    );
    assert!(!tensor_profile_by_name.contains_key("vision_tower.blocks.0.mlp.linear_fc2.scales"));
}

#[test]
fn should_reject_codebook_storage_on_vision_linear_modules() {
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(FROZEN_VISION_CONFIG_JSON.as_bytes())
        .expect("the frozen Ornith 1.0 vision config should parse");
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Qwen3.5 config should decode");
    let mut quantization_overrides = config_value["quantization"]
        .as_object()
        .expect("the frozen quantization contract should be an object")
        .clone();
    quantization_overrides.insert(
        "vision_tower.blocks.0.attn.qkv".to_owned(),
        json!({"mode": "jangtq2", "bits": 2, "rotation": "hadamard32"}),
    );
    let quantization_value = Value::Object(quantization_overrides);
    config_value["quantization"] = quantization_value.clone();
    config_value["quantization_config"] = quantization_value;
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the synthetic vision-codebook config should serialize");
    let qwen3_5_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the synthetic vision-codebook config should parse");

    let profile_result = qwen3_5_vision_tensor_profiles_for_model(&vision_config, &qwen3_5_config);

    assert!(matches!(
        profile_result,
        Err(crate::qwen3_5_core::artifacts::Qwen3_5ArtifactError::InvalidVisionQuantizationProfile {
            module_name,
            ..
        }) if module_name == "vision_tower.blocks.0.attn.qkv"
    ));
}

fn qwen_config_with_native_vision_mlp_output() -> Qwen3_5Config {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Qwen3.5 config should decode");
    let mut quantization_overrides = config_value["quantization"]
        .as_object()
        .expect("the frozen quantization contract should be an object")
        .clone();
    for vision_block_index in 0..27 {
        quantization_overrides.insert(
            format!("vision_tower.blocks.{vision_block_index}.mlp.linear_fc2"),
            Value::Bool(false),
        );
    }
    let quantization_value = Value::Object(quantization_overrides);
    config_value["quantization"] = quantization_value.clone();
    config_value["quantization_config"] = quantization_value;
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the synthetic affine-vision config should serialize");
    Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the synthetic affine-vision config should parse")
}
