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

        let config = Qwen3_5Config::from_json_bytes(&config_bytes)
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

        let config = Qwen3_5Config::from_json_bytes(&config_bytes)
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

    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
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
        Qwen3_5Config::from_json_bytes(&invalid_config_bytes),
        Err(Qwen3_5ConfigError::InvalidConfigValueDynamic { .. })
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
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
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
fn should_create_an_unquantized_quantization_profile() {
    use astronomical_model_serving::OptiQQuantizationProfile;

    let profile = OptiQQuantizationProfile::unquantized();
    assert_eq!(profile.bits, 0);
    assert_eq!(profile.group_size, 0);
    assert!(profile.is_unquantized());
}
