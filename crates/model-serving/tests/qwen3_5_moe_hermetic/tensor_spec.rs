use std::collections::BTreeMap;

use astronomical_model_serving::{
    TensorDtype, TensorProfile, qwen3_5_language_tensor_profiles,
    qwen3_5_resident_language_tensor_profiles,
};

use crate::common::qwen3_5_moe::frozen_ornith_1_0_config;

#[test]
fn should_generate_the_complete_mixed_precision_optiq_language_tensor_profile() {
    let ornith_config = frozen_ornith_1_0_config();

    let tensor_profiles = qwen3_5_language_tensor_profiles(&ornith_config);
    let tensor_profile_by_name = tensor_profile_by_name(&tensor_profiles);

    assert_eq!(tensor_profiles.len(), 1_757);
    assert_eq!(tensor_profile_by_name.len(), 1_757);
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.embed_tokens.weight",
        TensorDtype::UInt32,
        &[248_320, 512],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.embed_tokens.scales",
        TensorDtype::AffineQuantizationFloat,
        &[248_320, 32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        TensorDtype::ModelFloat,
        &[8_192, 4, 1],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        TensorDtype::UInt32,
        &[8_192, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_z.scales",
        TensorDtype::AffineQuantizationFloat,
        &[4_096, 32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.in_proj_b.biases",
        TensorDtype::AffineQuantizationFloat,
        &[32, 32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.A_log",
        TensorDtype::ModelFloat,
        &[32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.dt_bias",
        TensorDtype::ModelFloat,
        &[32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.linear_attn.norm.weight",
        TensorDtype::ModelFloat,
        &[128],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.gate.weight",
        TensorDtype::UInt32,
        &[256, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        TensorDtype::UInt32,
        &[256, 512, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales",
        TensorDtype::AffineQuantizationFloat,
        &[256, 2_048, 8],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        TensorDtype::UInt32,
        &[2_048, 64],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        TensorDtype::UInt32,
        &[1, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.q_proj.weight",
        TensorDtype::UInt32,
        &[8_192, 256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.39.self_attn.q_proj.weight",
        TensorDtype::UInt32,
        &[8_192, 512],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.o_proj.scales",
        TensorDtype::AffineQuantizationFloat,
        &[2_048, 64],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.3.self_attn.q_norm.weight",
        TensorDtype::ModelFloat,
        &[256],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.layers.39.self_attn.k_proj.biases",
        TensorDtype::AffineQuantizationFloat,
        &[512, 32],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.model.norm.weight",
        TensorDtype::ModelFloat,
        &[2_048],
    );
    assert_tensor(
        &tensor_profile_by_name,
        "language_model.lm_head.biases",
        TensorDtype::AffineQuantizationFloat,
        &[248_320, 32],
    );
    assert_eq!(count_tensors_in_layer(&tensor_profiles, 0), 45);
    assert_eq!(count_tensors_in_layer(&tensor_profiles, 3), 40);
    assert!(
        !tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.contains("in_proj_qkvz")
                || tensor_name.contains("in_proj_ba"))
    );
    assert!(
        !tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.starts_with("vision_tower."))
    );
}

#[test]
fn should_exclude_sparse_expert_tensors_from_every_resident_profile() {
    let ornith_config = frozen_ornith_1_0_config();

    let resident_tensor_profiles = qwen3_5_resident_language_tensor_profiles(&ornith_config);
    let resident_tensor_profile_by_name = tensor_profile_by_name(&resident_tensor_profiles);

    assert_eq!(resident_tensor_profiles.len(), 1_397);
    assert_eq!(count_tensors_in_layer(&resident_tensor_profiles, 0), 36);
    assert_eq!(count_tensors_in_layer(&resident_tensor_profiles, 3), 31);
    assert!(
        !resident_tensor_profile_by_name
            .keys()
            .any(|tensor_name| tensor_name.contains(".mlp.switch_mlp.")),
        "resident profile must not bind sparse selected experts"
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.gate.weight",
        TensorDtype::UInt32,
        &[256, 256],
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        TensorDtype::UInt32,
        &[2_048, 64],
    );
    assert_tensor(
        &resident_tensor_profile_by_name,
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        TensorDtype::UInt32,
        &[1, 256],
    );
}

fn tensor_profile_by_name(tensor_profiles: &[TensorProfile]) -> BTreeMap<&str, &TensorProfile> {
    tensor_profiles
        .iter()
        .map(|tensor_profile| (tensor_profile.name.as_str(), tensor_profile))
        .collect()
}

fn assert_tensor(
    tensor_profile_by_name: &BTreeMap<&str, &TensorProfile>,
    tensor_name: &str,
    expected_dtype: TensorDtype,
    expected_shape: &[usize],
) {
    let tensor_profile = tensor_profile_by_name
        .get(tensor_name)
        .unwrap_or_else(|| panic!("expected tensor profile {tensor_name}"));
    assert_eq!(
        tensor_profile.dtype, expected_dtype,
        "dtype for {tensor_name}"
    );
    assert_eq!(
        tensor_profile.shape, expected_shape,
        "shape for {tensor_name}"
    );
}

fn count_tensors_in_layer(tensor_profiles: &[TensorProfile], decoder_layer_index: usize) -> usize {
    let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}.");
    tensor_profiles
        .iter()
        .filter(|tensor_profile| tensor_profile.name.starts_with(&layer_prefix))
        .count()
}
