use super::*;

#[test]
fn should_parse_the_frozen_qwen3_5_moe_text_core_config() {
    let config_bytes = frozen_ornith_1_0_config_bytes();

    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 core config should parse");

    assert_eq!(ornith_config.hidden_size(), 2_048);
    assert_eq!(ornith_config.layer_count(), 40);
    assert_eq!(ornith_config.vocabulary_size(), 248_320);
    assert_eq!(ornith_config.mtp_layer_count(), 1);
    assert_eq!(ornith_config.torch_dtype(), "bfloat16");
    assert_eq!(ornith_config.hidden_activation(), "silu");
    assert_eq!(ornith_config.rms_norm_epsilon_bits(), 1e-6_f32.to_bits());
    assert_eq!(ornith_config.rope_theta_bits(), 10_000_000_f32.to_bits());
    assert_eq!(
        ornith_config.partial_rotary_factor_bits(),
        0.25_f32.to_bits()
    );
    assert_eq!(
        ornith_config.end_of_sequence_token_ids(),
        [248_046, 248_044]
    );
    assert!(!ornith_config.has_attention_bias());
    assert!(!ornith_config.has_mlp_bias());
    assert!(!ornith_config.has_tied_embeddings());
    assert!(ornith_config.normalizes_top_k_probabilities());
    assert_eq!(
        ornith_config.context_memory_reservation_bytes(1),
        Some(20_480),
        "request admission should reserve only context-growing full-attention KV state, not all 40 decoder-layer activations"
    );
}

#[test]
fn should_parse_a_native_bfloat16_config_without_quantization_metadata() {
    let mut native_bfloat16_config_document =
        serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
            .expect("the frozen Ornith 1.0 config should decode as JSON");
    native_bfloat16_config_document
        .as_object_mut()
        .expect("the frozen Ornith 1.0 config should be a JSON object")
        .remove("quantization");
    native_bfloat16_config_document
        .as_object_mut()
        .expect("the frozen Ornith 1.0 config should remain a JSON object")
        .remove("quantization_config");
    let native_bfloat16_config_bytes = serde_json::to_vec(&native_bfloat16_config_document)
        .expect("the native BF16 config should serialize");

    let native_bfloat16_config = Qwen3_5Config::from_json_bytes(&native_bfloat16_config_bytes)
        .expect("a native BF16 Qwen3.5-MoE config should not require quantization metadata");

    assert_eq!(native_bfloat16_config.activation_dtype(), "bfloat16");
    assert_eq!(
        native_bfloat16_config.model_weight_storage(),
        ModelWeightStorage::NativeBfloat16,
        "an absent quantization contract must explicitly identify native BF16 storage"
    );
    let native_bfloat16_tensor_profiles = qwen3_5_language_tensor_profiles(&native_bfloat16_config);
    assert!(
        native_bfloat16_tensor_profiles
            .iter()
            .any(|tensor_profile| tensor_profile.name == "language_model.lm_head.weight"),
        "native BF16 artifacts must retain their dense weight tensor"
    );
    assert!(
        !native_bfloat16_tensor_profiles
            .iter()
            .any(|tensor_profile| tensor_profile.name == "language_model.lm_head.scales"),
        "native BF16 artifacts must not require affine scales"
    );
}

#[test]
fn should_parse_a_config_with_only_one_quantization_document() {
    let mut single_quantization_document =
        serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
            .expect("the frozen Ornith 1.0 config should decode as JSON");
    single_quantization_document
        .as_object_mut()
        .expect("the frozen Ornith 1.0 config should be a JSON object")
        .remove("quantization_config");
    let single_quantization_bytes = serde_json::to_vec(&single_quantization_document)
        .expect("the single quantization config should serialize");

    let parsed_config = Qwen3_5Config::from_json_bytes(&single_quantization_bytes)
        .expect("one valid quantization document should be sufficient");

    assert_eq!(
        parsed_config.model_weight_storage(),
        ModelWeightStorage::AffineQuantized
    );
    assert_eq!(parsed_config.default_quantization_bits(), 6);
    assert_eq!(parsed_config.default_quantization_group_size(), 64);
}

#[test]
fn should_estimate_long_context_memory_from_full_attention_key_value_state_only() {
    let ornith_config = Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Ornith 1.0 core config should parse");

    assert_eq!(
        ornith_config.context_memory_reservation_bytes(179_350),
        Some(3_673_088_000),
        "the 179k-token OpenCode request should reserve about 3.4 GiB of KV state, not the inflated 29.4 GB all-layer activation estimate"
    );
}

#[test]
fn should_reject_a_context_memory_reservation_that_overflows_the_platform_range() {
    let ornith_config = Qwen3_5Config::from_json_bytes(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen Ornith 1.0 core config should parse");

    assert_eq!(
        ornith_config.context_memory_reservation_bytes(usize::MAX),
        None
    );
}

#[test]
fn should_use_the_declared_attention_type_for_each_decoder_layer() {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen test config should decode as JSON");
    config_value["text_config"]["layer_types"][0] = json!("full_attention");
    config_value["text_config"]["layer_types"][3] = json!("linear_attention");
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the declared per-layer attention schedule should parse");
    let tensor_profiles = qwen3_5_language_tensor_profiles(&config);
    let tensor_names = tensor_profiles
        .iter()
        .map(|tensor_profile| tensor_profile.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();

    assert!(tensor_names.contains("language_model.model.layers.0.self_attn.q_proj.weight"));
    assert!(tensor_names.contains("language_model.model.layers.3.linear_attn.in_proj_qkv.weight"));
}

#[test]
fn should_reject_an_ornith_layer_schedule_with_the_wrong_number_of_layers() {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_config_bytes())
        .expect("the frozen test config should decode as JSON");
    config_value["text_config"]["layer_types"] = json!([]);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the modified Ornith config should serialize as JSON");

    assert!(matches!(
        Qwen3_5Config::from_json_bytes(&config_bytes),
        Err(Qwen3_5ConfigError::LayerTypeCountMismatch {
            actual_layer_type_count: 0,
            expected_layer_type_count: 40,
        })
    ));
}

#[test]
fn should_reject_linear_attention_dimensions_that_exceed_the_mlx_shape_range() {
    let mut config_value = minimal_valid_config_json();
    config_value["text_config"]["linear_num_key_heads"] = json!(u32::MAX);
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the oversized linear-attention config should serialize as JSON");

    assert!(matches!(
        Qwen3_5Config::from_json_bytes(&config_bytes),
        Err(Qwen3_5ConfigError::InvalidConfigValue { .. })
            | Err(Qwen3_5ConfigError::InvalidConfigValueDynamic { .. })
    ));
}
