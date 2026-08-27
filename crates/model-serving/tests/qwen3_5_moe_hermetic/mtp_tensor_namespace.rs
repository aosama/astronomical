use astronomical_model_serving::{
    Qwen3_5Config, TensorDtype, TensorProfile, qwen3_5_mtp_tensor_names,
    qwen3_5_mtp_tensor_profiles,
};

use crate::common::qwen3_5_moe::certified_optiq_ornith_config_bytes;

fn assert_mtp_namespace_covers_required_modules(
    mtp_tensor_names: &std::collections::BTreeSet<String>,
) {
    for required_module_suffix in [
        "fc.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.up_proj.weight",
        "layers.0.mlp.shared_expert.down_proj.weight",
        "layers.0.mlp.shared_expert_gate.weight",
        "layers.0.mlp.switch_mlp.gate_proj.weight",
        "layers.0.mlp.switch_mlp.up_proj.weight",
        "layers.0.mlp.switch_mlp.down_proj.weight",
    ] {
        let required_tensor_name = format!("language_model.mtp.{required_module_suffix}");
        assert!(
            mtp_tensor_names.contains(&required_tensor_name),
            "missing required MTP tensor {required_tensor_name}"
        );
    }
}

fn assert_affine_module_is_native_or_packed(
    mtp_tensor_profiles: &[TensorProfile],
    module_name: &str,
) {
    let weight_tensor_name = format!("{module_name}.weight");
    let weight_profile = mtp_tensor_profiles
        .iter()
        .find(|tensor_profile| tensor_profile.name == weight_tensor_name)
        .unwrap_or_else(|| panic!("missing MTP weight {weight_tensor_name}"));
    let has_scales = mtp_tensor_profiles
        .iter()
        .any(|tensor_profile| tensor_profile.name == format!("{module_name}.scales"));
    let has_biases = mtp_tensor_profiles
        .iter()
        .any(|tensor_profile| tensor_profile.name == format!("{module_name}.biases"));
    if has_scales || has_biases {
        assert_eq!(weight_profile.dtype, TensorDtype::UInt32);
        assert!(has_scales && has_biases);
        assert!(!weight_profile.shape.is_empty());
    } else {
        assert_eq!(weight_profile.dtype, TensorDtype::ModelFloat);
        assert!(!weight_profile.shape.is_empty());
    }
}

#[test]
fn should_build_the_complete_one_layer_qwen_quantized_mtp_tensor_namespace() {
    let optiq_config = Qwen3_5Config::from_json_bytes(&certified_optiq_ornith_config_bytes())
        .expect("the OptiQ configuration should parse");
    let expected_tensor_names = qwen3_5_mtp_tensor_names(&optiq_config);

    assert!(!expected_tensor_names.is_empty());
    assert!(
        expected_tensor_names
            .iter()
            .all(|tensor_name| tensor_name.starts_with("language_model.mtp."))
    );
    assert_mtp_namespace_covers_required_modules(&expected_tensor_names);
}

#[test]
fn should_describe_quantized_or_native_mtp_affine_modules_from_config() {
    let optiq_config = Qwen3_5Config::from_json_bytes(&certified_optiq_ornith_config_bytes())
        .expect("the OptiQ configuration should parse");
    let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&optiq_config);

    assert!(!mtp_tensor_profiles.is_empty());
    for module_name in [
        "language_model.mtp.fc",
        "language_model.mtp.layers.0.self_attn.q_proj",
        "language_model.mtp.layers.0.mlp.gate",
        "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj",
        "language_model.mtp.layers.0.mlp.shared_expert.down_proj",
    ] {
        assert_affine_module_is_native_or_packed(&mtp_tensor_profiles, module_name);
    }
}

#[test]
fn should_describe_an_index_resolved_native_mtp_projection_without_affine_companions() {
    let mut optiq_config = Qwen3_5Config::from_json_bytes(&certified_optiq_ornith_config_bytes())
        .expect("the OptiQ configuration should parse");
    let native_mtp_module_name = "language_model.mtp.layers.0.self_attn.q_proj";
    let shard_tensor_names = [format!("{native_mtp_module_name}.weight")]
        .into_iter()
        .collect();

    optiq_config.resolve_unquantized_modules_from_shard_index(&shard_tensor_names);

    let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&optiq_config);
    assert_affine_module_is_native_or_packed(&mtp_tensor_profiles, native_mtp_module_name);
    assert!(
        !mtp_tensor_profiles
            .iter()
            .any(|tensor_profile| tensor_profile.name == format!("{native_mtp_module_name}.scales"))
    );
    assert!(
        !mtp_tensor_profiles
            .iter()
            .any(|tensor_profile| tensor_profile.name == format!("{native_mtp_module_name}.biases"))
    );
}
