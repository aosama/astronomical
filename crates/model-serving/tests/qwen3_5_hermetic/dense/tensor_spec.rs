use astronomical_model_serving::{TensorDtype, qwen3_5_language_tensor_profiles};

use crate::common::qwen3_5::certified_dense_qwen3_6_config;

#[test]
fn should_generate_dense_qwen3_5_language_tensor_profiles_without_sparse_experts() {
    let dense_qwen3_6_config = certified_dense_qwen3_6_config();
    let dense_tensor_profiles = qwen3_5_language_tensor_profiles(&dense_qwen3_6_config);
    let dense_intermediate_size = dense_qwen3_6_config.dense_intermediate_size() as usize;
    let hidden_size = dense_qwen3_6_config.hidden_size() as usize;

    for projection_name in ["gate_proj", "up_proj"] {
        assert!(dense_tensor_profiles.iter().any(|tensor_profile| {
            tensor_profile.name
                == format!("language_model.model.layers.0.mlp.{projection_name}.weight")
                && tensor_profile.dtype == TensorDtype::UInt32
                && tensor_profile.shape == [dense_intermediate_size, hidden_size / 8]
        }));
    }
    assert!(dense_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name == "language_model.model.layers.0.mlp.down_proj.weight"
            && tensor_profile.dtype == TensorDtype::UInt32
            && tensor_profile.shape == [hidden_size, dense_intermediate_size / 8]
    }));
    assert!(!dense_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name.contains(".mlp.switch_mlp.")
            || tensor_profile.name.contains(".mlp.gate.")
            || tensor_profile.name.contains(".mlp.shared_expert")
    }));
}
