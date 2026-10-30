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

// A deliberately minimal validated vision geometry: patch and merge boundary
// values of 1 so the derived widths (qkv=3*hidden, merger=hidden) are trivial
// to check without the full Ornith dimensions.
fn minimal_vision_config(block_depth: u32) -> Qwen3_5VisionConfig {
    let vision_config_json = json!({
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": block_depth,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 64,
            "in_channels": 3,
            "intermediate_size": 128,
            "model_type": "qwen3_5_vision",
            "num_heads": 4,
            "num_position_embeddings": 16,
            "out_hidden_size": 32,
            "patch_size": 1,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1
        }
    });
    Qwen3_5VisionConfig::from_json_bytes(vision_config_json.to_string().as_bytes())
        .expect("the minimal vision config should parse")
}

#[test]
fn should_derive_vision_geometry_dimensions_from_the_validated_config() {
    let vision_config = Qwen3_5VisionConfig::from_json_bytes(FROZEN_VISION_CONFIG_JSON.as_bytes())
        .expect("the frozen Ornith 1.0 vision config should parse");
    let geometry = VisionGeometry::from_config(&vision_config);
    assert_eq!(geometry.hidden_size, 1152);
    assert_eq!(geometry.intermediate_size, 4304);
    assert_eq!(geometry.depth, 27);
    assert_eq!(geometry.patch_size, 16);
    assert_eq!(geometry.temporal_patch_size, 2);
    assert_eq!(geometry.in_channels, 3);
    assert_eq!(geometry.position_embedding_count, 2304);
    assert_eq!(geometry.out_hidden_size, 2048);
    assert_eq!(geometry.qkv_dimension, 3456);
    assert_eq!(geometry.merger_input_dimension, 4608);
}

#[test]
fn should_keep_boundary_geometry_dimensions_for_the_smallest_valid_patch_and_merge() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    assert_eq!(geometry.patch_size, 1);
    assert_eq!(geometry.temporal_patch_size, 1);
    // spatial_merge_size 1 is the boundary where the merger input stops being
    // four concatenated hidden rows and collapses to exactly one hidden row,
    // guarding against a hardcoded 4x multiplication.
    assert_eq!(geometry.merger_input_dimension, geometry.hidden_size);
    assert_eq!(geometry.qkv_dimension, 192);
}

#[test]
fn should_append_patch_and_positional_embedding_tensor_profiles() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(2));
    let mut tensor_profiles = Vec::new();
    append_patch_embedding_tensor_profiles(&mut tensor_profiles, &geometry);
    append_positional_embedding_tensor_profile(&mut tensor_profiles, &geometry);
    assert_eq!(tensor_profiles.len(), 3);
    assert_eq!(
        tensor_profiles[0].name,
        "vision_tower.patch_embed.proj.weight"
    );
    assert_eq!(tensor_profiles[0].dtype, TensorDtype::ModelFloat);
    assert_eq!(tensor_profiles[0].shape, vec![64, 1, 1, 1, 3]);
    assert_eq!(
        tensor_profiles[0].equivalent_published_shapes,
        vec![vec![64, 3, 1, 1, 1]]
    );
    assert_eq!(
        tensor_profiles[1].name,
        "vision_tower.patch_embed.proj.bias"
    );
    assert_eq!(tensor_profiles[1].shape, vec![64]);
    assert_eq!(tensor_profiles[2].name, "vision_tower.pos_embed.weight");
    assert_eq!(tensor_profiles[2].shape, vec![16, 64]);
    for tensor_profile in &tensor_profiles {
        assert!(
            !tensor_profile.name.starts_with("vision_tower.blocks.")
                && !tensor_profile.name.starts_with("vision_tower.merger."),
            "embedding append must not leak {}",
            tensor_profile.name
        );
    }
}

#[test]
fn should_append_twelve_tensor_profiles_per_transformer_block() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(2));
    let mut tensor_profiles = Vec::new();
    append_transformer_block_tensor_profiles(&mut tensor_profiles, None, &geometry)
        .expect("native block profiles use the validated vision geometry");
    assert_eq!(tensor_profiles.len(), 24);
    let expected_block_zero_names = [
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.blocks.0.attn.qkv.bias",
        "vision_tower.blocks.0.attn.proj.weight",
        "vision_tower.blocks.0.attn.proj.bias",
        "vision_tower.blocks.0.norm1.weight",
        "vision_tower.blocks.0.norm1.bias",
        "vision_tower.blocks.0.mlp.linear_fc1.weight",
        "vision_tower.blocks.0.mlp.linear_fc1.bias",
        "vision_tower.blocks.0.mlp.linear_fc2.weight",
        "vision_tower.blocks.0.mlp.linear_fc2.bias",
        "vision_tower.blocks.0.norm2.weight",
        "vision_tower.blocks.0.norm2.bias",
    ];
    for (tensor_index, expected_name) in expected_block_zero_names.iter().enumerate() {
        assert_eq!(tensor_profiles[tensor_index].name, *expected_name);
    }
    assert_eq!(tensor_profiles[23].name, "vision_tower.blocks.1.norm2.bias");
    assert_eq!(tensor_profiles[0].shape, vec![192, 64]);
    assert_eq!(tensor_profiles[7].shape, vec![128]);
    assert_eq!(tensor_profiles[8].shape, vec![64, 128]);
    assert_eq!(tensor_profiles[9].shape, vec![64]);
}

#[test]
fn should_append_twelve_tensor_profiles_for_a_single_block_boundary() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut tensor_profiles = Vec::new();
    append_transformer_block_tensor_profiles(&mut tensor_profiles, None, &geometry)
        .expect("native block profiles use the validated vision geometry");
    assert_eq!(tensor_profiles.len(), 12);
    assert_eq!(tensor_profiles[11].name, "vision_tower.blocks.0.norm2.bias");
}

#[test]
fn should_append_affine_quantized_block_tensor_profiles_from_module_profiles() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut module_quantization_profiles = HashMap::new();
    module_quantization_profiles.insert(
        "vision_tower.blocks.0.attn.qkv".to_owned(),
        OptiQQuantizationProfile::affine(4, 64),
    );
    let mut tensor_profiles = Vec::new();
    append_transformer_block_tensor_profiles(
        &mut tensor_profiles,
        Some(&module_quantization_profiles),
        &geometry,
    )
    .expect("affine block profiles use the validated vision geometry");
    // qkv contributes weight + scales + biases + bias (4), the other five
    // modules contribute weight + bias each (10): 14 tensors for the block.
    assert_eq!(tensor_profiles.len(), 14);
    assert_eq!(tensor_profiles[0].dtype, TensorDtype::UInt32);
    // Packed width = 64 input values * 4 bits / 32 bits per word = 8.
    assert_eq!(tensor_profiles[0].shape, vec![192, 8]);
    assert_eq!(
        tensor_profiles[1].name,
        "vision_tower.blocks.0.attn.qkv.scales"
    );
    assert_eq!(
        tensor_profiles[1].dtype,
        TensorDtype::AffineQuantizationFloat
    );
    assert_eq!(tensor_profiles[1].shape, vec![192, 1]);
    assert_eq!(
        tensor_profiles[2].name,
        "vision_tower.blocks.0.attn.qkv.biases"
    );
    assert_eq!(
        tensor_profiles[3].name,
        "vision_tower.blocks.0.attn.qkv.bias"
    );
    assert_eq!(tensor_profiles[3].dtype, TensorDtype::ModelFloat);
    assert_eq!(tensor_profiles[3].shape, vec![192]);
    // Modules absent from the profile map must stay unquantized model floats.
    assert_eq!(
        tensor_profiles[4].name,
        "vision_tower.blocks.0.attn.proj.weight"
    );
    assert_eq!(tensor_profiles[4].dtype, TensorDtype::ModelFloat);
    assert_eq!(tensor_profiles[4].shape, vec![64, 64]);
}

#[test]
fn should_reject_codebook_quantized_module_during_block_profile_append() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut module_quantization_profiles = HashMap::new();
    module_quantization_profiles.insert(
        "vision_tower.blocks.0.attn.qkv".to_owned(),
        OptiQQuantizationProfile::jangtq2_codebook(
            2,
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::Hadamard32,
        ),
    );
    let profile_result = append_transformer_block_tensor_profiles(
        &mut Vec::new(),
        Some(&module_quantization_profiles),
        &geometry,
    );
    assert!(matches!(
        profile_result,
        Err(Qwen3_5ArtifactError::InvalidVisionQuantizationProfile {
            module_name,
            ..
        }) if module_name == "vision_tower.blocks.0.attn.qkv"
    ));
}

#[test]
fn should_append_merger_tensor_profiles() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut tensor_profiles = Vec::new();
    append_merger_tensor_profiles(&mut tensor_profiles, None, &geometry)
        .expect("native merger profiles use the validated vision geometry");
    assert_eq!(tensor_profiles.len(), 6);
    assert_eq!(tensor_profiles[0].name, "vision_tower.merger.norm.weight");
    assert_eq!(tensor_profiles[0].shape, vec![64]);
    assert_eq!(tensor_profiles[1].name, "vision_tower.merger.norm.bias");
    assert_eq!(
        tensor_profiles[2].name,
        "vision_tower.merger.linear_fc1.weight"
    );
    assert_eq!(tensor_profiles[2].shape, vec![64, 64]);
    assert_eq!(
        tensor_profiles[4].name,
        "vision_tower.merger.linear_fc2.weight"
    );
    assert_eq!(tensor_profiles[4].shape, vec![32, 64]);
    assert_eq!(
        tensor_profiles[5].name,
        "vision_tower.merger.linear_fc2.bias"
    );
    assert_eq!(tensor_profiles[5].shape, vec![32]);
}

#[test]
fn should_append_affine_quantized_merger_fc1_companions() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut module_quantization_profiles = HashMap::new();
    module_quantization_profiles.insert(
        "vision_tower.merger.linear_fc1".to_owned(),
        OptiQQuantizationProfile::affine(4, 64),
    );
    let mut tensor_profiles = Vec::new();
    append_merger_tensor_profiles(
        &mut tensor_profiles,
        Some(&module_quantization_profiles),
        &geometry,
    )
    .expect("affine merger profiles use the validated vision geometry");
    assert_eq!(tensor_profiles.len(), 8);
    assert_eq!(tensor_profiles[2].dtype, TensorDtype::UInt32);
    assert_eq!(tensor_profiles[2].shape, vec![64, 8]);
    assert_eq!(
        tensor_profiles[3].name,
        "vision_tower.merger.linear_fc1.scales"
    );
    assert_eq!(tensor_profiles[3].shape, vec![64, 1]);
    assert_eq!(
        tensor_profiles[4].name,
        "vision_tower.merger.linear_fc1.biases"
    );
    assert_eq!(
        tensor_profiles[5].name,
        "vision_tower.merger.linear_fc1.bias"
    );
    assert_eq!(tensor_profiles[5].dtype, TensorDtype::ModelFloat);
}

#[test]
fn should_reject_codebook_quantized_merger_module() {
    let geometry = VisionGeometry::from_config(&minimal_vision_config(1));
    let mut module_quantization_profiles = HashMap::new();
    module_quantization_profiles.insert(
        "vision_tower.merger.linear_fc2".to_owned(),
        OptiQQuantizationProfile::jangtq2_codebook(
            2,
            astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookRotation::NoRotation,
        ),
    );
    let profile_result = append_merger_tensor_profiles(
        &mut Vec::new(),
        Some(&module_quantization_profiles),
        &geometry,
    );
    assert!(matches!(
        profile_result,
        Err(Qwen3_5ArtifactError::InvalidVisionQuantizationProfile {
            module_name,
            ..
        }) if module_name == "vision_tower.merger.linear_fc2"
    ));
}
