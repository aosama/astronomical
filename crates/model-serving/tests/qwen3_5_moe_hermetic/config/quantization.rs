use std::collections::BTreeSet;

use astronomical_model_serving::{Qwen3_5Config, Qwen3_5ConfigError};
use serde_json::{Value, json};

use crate::common::qwen3_5_moe::{
    frozen_ornith_1_0_config_bytes, frozen_ornith_1_0_optiq_config_bytes,
    frozen_qwen3_8_distill_config_bytes,
};

#[test]
fn should_reject_a_router_gate_quantization_override_with_invalid_bits() {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen test config should decode as JSON");
    config_value["quantization"]["language_model.model.layers.0.mlp.gate"]["bits"] = json!(7);
    config_value["quantization_config"] = config_value["quantization"].clone();
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(matches!(
        Qwen3_5Config::from_json_bytes(&config_bytes),
        Err(Qwen3_5ConfigError::UnsupportedQuantizationOverrideBits { .. })
    ));
}

#[test]
fn should_parse_a_sparse_mixture_of_experts_quantization_config_with_mixed_group_sizes() {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen test config should decode as JSON");
    let mut quantization = json!({"group_size": 64, "bits": 6, "mode": "affine"});
    quantization["language_model.model.embed_tokens"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.model.layers.0.mlp.shared_expert_gate"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.model.layers.0.mlp.shared_expert.down_proj"] =
        json!({"group_size": 128, "bits": 8, "mode": "affine"});
    quantization["language_model.model.layers.0.linear_attn.out_proj"] =
        json!({"group_size": 128, "bits": 6, "mode": "affine"});
    quantization["language_model.model.layers.0.linear_attn.in_proj_qkv"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;

    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the sparse mixed-precision config should serialize as JSON");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the sparse mixed-precision config should parse");

    assert_eq!(config.default_quantization_bits(), 6);
    assert_eq!(config.default_quantization_group_size(), 64);
    let embed_profile = config.quantization_profile_for_module("language_model.model.embed_tokens");
    assert_eq!(embed_profile.bits, 8);
    assert_eq!(embed_profile.group_size, 64);
    let shared_down_profile = config.quantization_profile_for_module(
        "language_model.model.layers.0.mlp.shared_expert.down_proj",
    );
    assert_eq!(shared_down_profile.bits, 8);
    assert_eq!(shared_down_profile.group_size, 128);
    let out_proj_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.linear_attn.out_proj");
    assert_eq!(out_proj_profile.bits, 6);
    assert_eq!(out_proj_profile.group_size, 128);
    let switch_mlp_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.mlp.switch_mlp.gate_proj");
    assert_eq!(switch_mlp_profile.bits, 6);
    assert_eq!(switch_mlp_profile.group_size, 64);
}

#[test]
fn should_resolve_unquantized_modules_from_shard_index_when_scales_are_absent() {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen test config should decode as JSON");
    let mut quantization = json!({"group_size": 64, "bits": 6, "mode": "affine"});
    quantization["language_model.model.embed_tokens"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8, "mode": "affine"});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;

    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the sparse mixed-precision config should serialize as JSON");
    let mut config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the sparse mixed-precision config should parse");
    let gate_before =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert_eq!(gate_before.bits, 6);

    let mut shard_tensor_names = BTreeSet::new();
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales".to_owned());
    config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);

    let gate_after =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert!(gate_after.is_unquantized());
    let switch_mlp_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.mlp.switch_mlp.gate_proj");
    assert_eq!(switch_mlp_profile.bits, 6);
}

#[test]
fn should_not_resolve_gates_as_unquantized_when_scales_are_present() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let mut config =
        Qwen3_5Config::from_json_bytes(&config_bytes).expect("the oQ4 config should parse");

    let mut shard_tensor_names = BTreeSet::new();
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.weight".to_owned());
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.scales".to_owned());
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.biases".to_owned());
    config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);

    let gate_profile =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert!(!gate_profile.is_unquantized());
    assert_eq!(gate_profile.bits, 4);
}

#[test]
fn should_not_hide_missing_scales_when_affine_biases_are_present() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let mut config =
        Qwen3_5Config::from_json_bytes(&config_bytes).expect("the oQ4 config should parse");
    let shard_tensor_names = [
        "language_model.model.layers.0.mlp.gate.weight".to_owned(),
        "language_model.model.layers.0.mlp.gate.biases".to_owned(),
    ]
    .into_iter()
    .collect();

    config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);

    let gate_profile =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert!(!gate_profile.is_unquantized());
}

#[test]
fn should_parse_the_qwen_3_8_distill_sparse_mixed_precision_quantization_config() {
    let config_bytes = frozen_qwen3_8_distill_config_bytes();
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Qwen 3.8 distill config should parse");

    assert_eq!(config.default_quantization_bits(), 6);
    assert_eq!(config.default_quantization_group_size(), 64);
    let module_profile_expectations = [
        ("language_model.model.embed_tokens", 8, 64),
        ("language_model.lm_head", 8, 64),
        (
            "language_model.model.layers.0.mlp.shared_expert.gate_proj",
            8,
            128,
        ),
        (
            "language_model.model.layers.0.mlp.shared_expert_gate",
            8,
            64,
        ),
        ("language_model.model.layers.0.linear_attn.out_proj", 6, 128),
        (
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            8,
            64,
        ),
        (
            "language_model.model.layers.1.linear_attn.in_proj_qkv",
            6,
            64,
        ),
        ("language_model.model.layers.3.self_attn.q_proj", 8, 64),
        ("language_model.model.layers.11.self_attn.q_proj", 6, 64),
        (
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
            6,
            64,
        ),
    ];
    for (module_name, expected_bits, expected_group_size) in module_profile_expectations {
        let module_profile = config.quantization_profile_for_module(module_name);
        assert_eq!(module_profile.bits, expected_bits, "bits for {module_name}");
        assert_eq!(
            module_profile.group_size, expected_group_size,
            "group size for {module_name}"
        );
    }
}

#[test]
fn should_resolve_the_qwen_3_8_distill_router_and_mtp_fusion_as_unquantized() {
    let config_bytes = frozen_qwen3_8_distill_config_bytes();
    let mut config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Qwen 3.8 distill config should parse");

    // The artifact stores the router gates, the MTP fusion projection, and the
    // normalization weights as plain native tensors with no scales or biases.
    let mut shard_tensor_names = BTreeSet::new();
    for decoder_layer_index in 0..40 {
        shard_tensor_names.insert(format!(
            "language_model.model.layers.{decoder_layer_index}.mlp.gate.weight"
        ));
    }
    shard_tensor_names.insert("language_model.mtp.fc.weight".to_owned());
    shard_tensor_names.insert("language_model.mtp.norm.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases".to_owned());
    config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);

    for unquantized_module_name in [
        "language_model.model.layers.0.mlp.gate",
        "language_model.model.layers.39.mlp.gate",
        "language_model.mtp.fc",
        "language_model.mtp.norm",
    ] {
        let module_profile = config.quantization_profile_for_module(unquantized_module_name);
        assert!(
            module_profile.is_unquantized(),
            "{unquantized_module_name} should resolve as native floating point"
        );
    }
    let switch_mlp_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.mlp.switch_mlp.gate_proj");
    assert_eq!(switch_mlp_profile.bits, 6);
    assert_eq!(switch_mlp_profile.group_size, 64);
}

#[test]
fn should_resolve_qwen_3_8_distill_mtp_override_profiles() {
    let config_bytes = frozen_qwen3_8_distill_config_bytes();
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Qwen 3.8 distill config should parse");

    let mtp_profile_expectations = [
        ("language_model.mtp.layers.0.self_attn.q_proj", 8, 64),
        ("language_model.mtp.layers.0.self_attn.o_proj", 8, 64),
        (
            "language_model.mtp.layers.0.mlp.shared_expert.gate_proj",
            8,
            128,
        ),
        ("language_model.mtp.layers.0.mlp.shared_expert_gate", 8, 64),
        (
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj",
            6,
            64,
        ),
    ];
    for (module_name, expected_bits, expected_group_size) in mtp_profile_expectations {
        let module_profile = config.quantization_profile_for_module(module_name);
        assert_eq!(module_profile.bits, expected_bits, "bits for {module_name}");
        assert_eq!(
            module_profile.group_size, expected_group_size,
            "group size for {module_name}"
        );
    }
}
