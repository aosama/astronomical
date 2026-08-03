use astronomical_model_serving::{
    TensorDtype, qwen3_5_mtp_tensor_profiles, qwen3_5_quantized_mtp_tensor_names,
};

use crate::common::qwen3_5::certified_dense_qwen3_6_config;

#[test]
fn should_describe_the_dense_qwen3_6_mtp_tensor_namespace_and_shapes() {
    let dense_qwen3_6_config = certified_dense_qwen3_6_config();
    let expected_tensor_names = qwen3_5_quantized_mtp_tensor_names(&dense_qwen3_6_config);
    let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&dense_qwen3_6_config);

    assert_eq!(expected_tensor_names.len(), 29);
    assert!(expected_tensor_names.contains("language_model.mtp.layers.0.mlp.down_proj.biases"));
    assert!(!expected_tensor_names.contains("language_model.mtp.layers.0.mlp.gate.weight"));
    assert_eq!(mtp_tensor_profiles.len(), 29);
    assert!(mtp_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name == "language_model.mtp.layers.0.mlp.down_proj.weight"
            && tensor_profile.dtype == TensorDtype::UInt32
            && tensor_profile.shape == [2_048, 64]
    }));
}
