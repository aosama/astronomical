use astronomical_model_serving::{
    TensorDtype, TensorProfile, qwen3_5_mtp_tensor_names, qwen3_5_mtp_tensor_profiles,
};

use crate::common::qwen3_5::certified_dense_qwen3_6_config;

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
fn should_describe_the_dense_qwen3_6_mtp_tensor_namespace_and_shapes() {
    let dense_qwen3_6_config = certified_dense_qwen3_6_config();
    let expected_tensor_names = qwen3_5_mtp_tensor_names(&dense_qwen3_6_config);
    let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&dense_qwen3_6_config);

    assert!(!expected_tensor_names.is_empty());
    assert!(expected_tensor_names.contains("language_model.mtp.fc.weight"));
    assert!(expected_tensor_names.contains("language_model.mtp.layers.0.mlp.down_proj.weight"));
    assert!(!expected_tensor_names.contains("language_model.mtp.layers.0.mlp.gate.weight"));
    assert_eq!(expected_tensor_names.len(), mtp_tensor_profiles.len());
    assert_affine_module_is_native_or_packed(&mtp_tensor_profiles, "language_model.mtp.fc");
    assert_affine_module_is_native_or_packed(
        &mtp_tensor_profiles,
        "language_model.mtp.layers.0.mlp.down_proj",
    );
}
