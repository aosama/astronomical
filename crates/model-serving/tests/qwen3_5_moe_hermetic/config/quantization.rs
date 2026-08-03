use super::*;

#[test]
fn should_accept_every_affine_quantization_bit_width_supported_by_mlx() {
    for quantization_bits in [2, 3, 4, 5, 6, 8] {
        let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
            .expect("the certified test config should decode as JSON");
        config_value["quantization"]["bits"] = json!(quantization_bits);
        config_value["quantization_config"] = config_value["quantization"].clone();
        let config_bytes = serde_json::to_vec(&config_value)
            .expect("the modified Ornith config should serialize as JSON");

        let config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
            .expect("MLX-supported affine quantization bits should parse");
        assert_eq!(config.default_quantization_bits(), quantization_bits);
    }
}

#[test]
fn should_accept_every_affine_quantization_group_size_supported_by_mlx() {
    for quantization_group_size in [32, 64, 128] {
        let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
            .expect("the certified test config should decode as JSON");
        config_value["quantization"]["group_size"] = json!(quantization_group_size);
        config_value["quantization_config"] = config_value["quantization"].clone();
        let config_bytes = serde_json::to_vec(&config_value)
            .expect("the modified Ornith config should serialize as JSON");

        let config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
            .expect("MLX-supported affine quantization group sizes should parse");
        assert_eq!(
            config.default_quantization_group_size(),
            quantization_group_size
        );
    }
}

#[test]
fn should_retain_an_artifact_declared_mtp_quantization_override() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    let mtp_dense_projection_module_name = "language_model.mtp.layers.0.mlp.down_proj";
    config_value["quantization"][mtp_dense_projection_module_name] =
        json!({"bits": 5, "group_size": 32});
    config_value["quantization_config"] = config_value["quantization"].clone();
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    let config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
        .expect("the configuration with an MTP override should parse");

    assert_eq!(
        (
            config
                .quantization_profile_for_module(mtp_dense_projection_module_name)
                .bits,
            config
                .quantization_profile_for_module(mtp_dense_projection_module_name)
                .group_size,
        ),
        (5, 32)
    );
}

#[test]
fn should_reject_the_affine_quantization_bit_width_unsupported_by_mlx() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["quantization"]["bits"] = json!(7);
    config_value["quantization_config"] = config_value["quantization"].clone();
    let invalid_config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(matches!(
        Qwen3_5MoEConfig::from_json_bytes(&invalid_config_bytes),
        Err(Qwen3_5MoEConfigError::InvalidConfigValueDynamic { .. })
    ));
}

#[test]
fn should_reject_a_router_gate_quantization_override_with_invalid_bits() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    config_value["quantization"]["language_model.model.layers.0.mlp.gate"]["bits"] = json!(7);
    config_value["quantization_config"] = config_value["quantization"].clone();
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(matches!(
        Qwen3_5MoEConfig::from_json_bytes(&config_bytes),
        Err(Qwen3_5MoEConfigError::UnsupportedQuantizationOverrideBits { .. })
    ));
}

#[test]
fn should_parse_a_standard_six_bit_config_without_high_bit_embedding_overrides() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    let quantization = config_value["quantization"]
        .as_object_mut()
        .expect("the quantization config should be an object");
    quantization.remove("language_model.model.embed_tokens");
    quantization.remove("language_model.lm_head");
    config_value["quantization_config"] = config_value["quantization"].clone();

    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the standard six-bit config should serialize as JSON");
    let config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
        .expect("the standard six-bit config should parse");

    assert_eq!(
        config
            .quantization_profile_for_module("language_model.model.embed_tokens")
            .bits,
        6
    );
    assert_eq!(
        config
            .quantization_profile_for_module("language_model.lm_head")
            .bits,
        6
    );
}

#[test]
fn should_parse_an_oq6e_sparse_quantization_config_with_mixed_group_sizes() {
    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    // Build an oQ6e-style config: default bits=6, gs=64 with sparse overrides
    // and mixed group sizes (64 and 128).
    let mut quantization = json!({
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    });
    // Embed tokens and lm_head must be 8-bit (minimum for embedding/lm_head)
    quantization["language_model.model.embed_tokens"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8, "mode": "affine"});
    // Shared expert gate at 8-bit with gs=64
    quantization["language_model.model.layers.0.mlp.shared_expert_gate"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    // Shared expert down_proj at 8-bit with gs=128 (mixed group size)
    quantization["language_model.model.layers.0.mlp.shared_expert.down_proj"] =
        json!({"group_size": 128, "bits": 8, "mode": "affine"});
    // Linear attention out_proj at 6-bit with gs=128 (mixed group size)
    quantization["language_model.model.layers.0.linear_attn.out_proj"] =
        json!({"group_size": 128, "bits": 6, "mode": "affine"});
    // Linear attention in_proj at 8-bit with gs=64
    quantization["language_model.model.layers.0.linear_attn.in_proj_qkv"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the oQ6e-style config should serialize as JSON");
    let config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
        .expect("the oQ6e-style sparse config should parse");

    // Default should be bits=6, gs=64
    assert_eq!(config.default_quantization_bits(), 6);
    assert_eq!(config.default_quantization_group_size(), 64);

    // Explicit overrides should have correct profiles
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

    // Modules NOT in overrides should get the default
    let switch_mlp_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.mlp.switch_mlp.gate_proj");
    assert_eq!(switch_mlp_profile.bits, 6);
    assert_eq!(switch_mlp_profile.group_size, 64);
}

#[test]
fn should_resolve_unquantized_gates_from_shard_index_when_scales_are_absent() {
    use std::collections::BTreeSet;

    let mut config_value = serde_json::from_slice::<Value>(&certified_ornith_config_bytes())
        .expect("the certified test config should decode as JSON");
    // Build an oQ6e-style config with default bits=6
    let mut quantization = json!({
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    });
    quantization["language_model.model.embed_tokens"] =
        json!({"group_size": 64, "bits": 8, "mode": "affine"});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8, "mode": "affine"});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;

    let config_bytes =
        serde_json::to_vec(&config_value).expect("the oQ6e-style config should serialize as JSON");
    let mut config = Qwen3_5MoEConfig::from_json_bytes(&config_bytes)
        .expect("the oQ6e-style config should parse");

    // Before resolution, the gate should have default bits=6
    let gate_before =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert_eq!(
        gate_before.bits, 6,
        "gate should initially have default bits=6"
    );

    // Build a shard tensor name set WITHOUT gate.scales
    let mut shard_tensor_names = BTreeSet::new();
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight".to_owned());
    shard_tensor_names
        .insert("language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales".to_owned());

    config.resolve_unquantized_gates_from_shard_index(&shard_tensor_names);

    // After resolution, the gate should be unquantized (bits=0)
    let gate_after =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert!(
        gate_after.is_unquantized(),
        "gate should be unquantized after resolving from shard index (got bits={}, gs={})",
        gate_after.bits,
        gate_after.group_size,
    );

    // switch_mlp.gate_proj should still be quantized (it has .scales)
    let switch_mlp_profile = config
        .quantization_profile_for_module("language_model.model.layers.0.mlp.switch_mlp.gate_proj");
    assert_eq!(
        switch_mlp_profile.bits, 6,
        "switch_mlp.gate_proj should still have default bits=6"
    );
}

#[test]
fn should_not_resolve_gates_as_unquantized_when_scales_are_present() {
    use std::collections::BTreeSet;

    // Use the oQ4 config which has explicit gate overrides (bits=4 for layer 0)
    let config_bytes = certified_optiq_ornith_config_bytes();
    let mut config =
        Qwen3_5MoEConfig::from_json_bytes(&config_bytes).expect("the oQ4 config should parse");

    // Build a shard tensor name set WITH gate.scales (oQ4 style)
    let mut shard_tensor_names = BTreeSet::new();
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.weight".to_owned());
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.scales".to_owned());
    shard_tensor_names.insert("language_model.model.layers.0.mlp.gate.biases".to_owned());

    config.resolve_unquantized_gates_from_shard_index(&shard_tensor_names);

    // Gate should remain quantized (bits=4 in oQ4 layer 0) — not converted to unquantized
    let gate_profile =
        config.quantization_profile_for_module("language_model.model.layers.0.mlp.gate");
    assert!(
        !gate_profile.is_unquantized(),
        "oQ4 gate should remain quantized when scales are present in the shard index"
    );
    assert_eq!(
        gate_profile.bits, 4,
        "oQ4 gate should keep its original 4-bit quantization"
    );
}

#[test]
fn should_not_add_sparse_router_gate_profiles_when_resolving_a_dense_model() {
    use std::collections::BTreeSet;

    let mut dense_config_document = minimal_valid_config_json();
    dense_config_document["architectures"] = json!(["Qwen3_5ForConditionalGeneration"]);
    dense_config_document["model_type"] = json!("qwen3_5");
    dense_config_document["text_config"]["model_type"] = json!("qwen3_5_text");
    dense_config_document["text_config"]["num_experts"] = json!(0);
    dense_config_document["text_config"]["num_experts_per_tok"] = json!(0);
    dense_config_document["text_config"]["moe_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["shared_expert_intermediate_size"] = json!(0);
    dense_config_document["text_config"]["intermediate_size"] = json!(512);
    let dense_config_bytes = serde_json::to_vec(&dense_config_document)
        .expect("the dense test config should serialize as JSON");
    let mut dense_config = Qwen3_5MoEConfig::from_json_bytes(&dense_config_bytes)
        .expect("the dense test config should parse");

    dense_config.resolve_unquantized_gates_from_shard_index(&BTreeSet::new());

    assert!(
        !dense_config
            .quantized_module_profiles()
            .contains_key("language_model.model.layers.0.mlp.gate"),
        "dense models do not have sparse router gates"
    );
}

#[test]
fn should_create_an_unquantized_quantization_profile() {
    use astronomical_model_serving::OptiQQuantizationProfile;

    let profile = OptiQQuantizationProfile::unquantized();
    assert_eq!(profile.bits, 0);
    assert_eq!(profile.group_size, 0);
    assert!(profile.is_unquantized());
}
