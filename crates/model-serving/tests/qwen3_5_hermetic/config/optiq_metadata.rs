use super::*;

#[test]
fn should_accept_two_bit_optiq_metadata_supported_by_mlx_affine_quantization() {
    let optiq_metadata_bytes = serde_json::to_vec(&json!({
        "method": "static_mixed_precision",
        "base_model": "mlx-community/Qwen3.5-122B-A10B-bf16",
        "reference": "structural_rules",
        "target_bpw": 2.5,
        "achieved_bpw": 2.5,
        "n_high_bits": 0,
        "n_low_bits": 1,
        "threshold": 0.0,
        "per_layer": {
            "language_model.model.layers.5.mlp.switch_mlp.gate_proj": {
                "bits": 2,
                "group_size": 64
            }
        }
    }))
    .expect("the two-bit OptiQ metadata fixture should serialize");

    let optiq_metadata = OptiQMetadata::from_json_bytes(&optiq_metadata_bytes)
        .expect("two-bit OptiQ metadata should use an MLX-supported affine bit width");

    assert_eq!(optiq_metadata.measured_module_count(), 1);
}

#[test]
fn should_require_the_optiq_metadata_bit_map_to_match_the_config() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let metadata_bytes = frozen_optiq_metadata_bytes();
    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 OptiQ config should parse");

    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the frozen OptiQ metadata should parse");

    assert_eq!(optiq_metadata.measured_module_count(), 510);
    optiq_metadata
        .validate_against_config(&ornith_config)
        .expect("the measured OptiQ bit map should exactly match the config overrides");
}

#[test]
fn should_accept_measured_optiq_profiles_that_are_a_strict_subset_of_the_config() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 OptiQ config should parse");
    let mut metadata_document = serde_json::from_slice::<Value>(&frozen_optiq_metadata_bytes())
        .expect("the frozen OptiQ metadata should decode as JSON");
    let measured_module_profiles = metadata_document["per_layer"]
        .as_object_mut()
        .expect("the frozen OptiQ metadata should contain measured module profiles");
    measured_module_profiles.retain(|module_name, _| !module_name.contains(".mlp.switch_mlp."));
    let metadata_bytes = serde_json::to_vec(&metadata_document)
        .expect("the measured-subset OptiQ metadata should serialize");

    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the measured-subset OptiQ metadata should parse");

    assert_eq!(optiq_metadata.measured_module_count(), 390);
    optiq_metadata
        .validate_against_config(&ornith_config)
        .expect("every declared OptiQ measurement should match the config profile");
}

#[test]
fn should_accept_optional_output_head_measurement_in_optiq_metadata() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 OptiQ config should parse");
    let mut metadata_document = serde_json::from_slice::<Value>(&frozen_optiq_metadata_bytes())
        .expect("the frozen OptiQ metadata should decode as JSON");
    metadata_document["per_layer"]["language_model.lm_head"] = json!({
        "bits": 8,
        "group_size": 64
    });
    let metadata_bytes = serde_json::to_vec(&metadata_document)
        .expect("the output-head OptiQ metadata should serialize");

    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("the output-head OptiQ metadata should parse");

    optiq_metadata
        .validate_against_config(&ornith_config)
        .expect("an optional output-head measurement should match its config profile");
}

#[test]
fn should_compare_supported_optiq_metadata_group_sizes_with_the_model_config() {
    let config_bytes = frozen_ornith_1_0_optiq_config_bytes();
    let ornith_config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the frozen Ornith 1.0 OptiQ config should parse");
    let mut metadata_document = serde_json::from_slice::<Value>(&frozen_optiq_metadata_bytes())
        .expect("the frozen OptiQ metadata should decode as JSON");
    let measured_module_profiles = metadata_document["per_layer"]
        .as_object_mut()
        .expect("the frozen OptiQ metadata should contain measured module profiles");
    let (_, first_measured_module_profile) = measured_module_profiles
        .iter_mut()
        .next()
        .expect("the frozen OptiQ metadata should measure at least one module");
    first_measured_module_profile["group_size"] = json!(32);
    let metadata_bytes = serde_json::to_vec(&metadata_document)
        .expect("the modified OptiQ metadata should serialize");

    let optiq_metadata = OptiQMetadata::from_json_bytes(&metadata_bytes)
        .expect("MLX-supported OptiQ group sizes should parse");

    let validation_error = optiq_metadata
        .validate_against_config(&ornith_config)
        .expect_err("a metadata group size that differs from config should fail validation");
    assert!(matches!(
        validation_error,
        OptiQMetadataError::ConfigGroupSizeMismatch {
            config_group_size: 64,
            metadata_group_size: 32,
            ..
        }
    ));
}
