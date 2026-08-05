use astronomical_model_serving::{Qwen3_5Config, TensorDtype, qwen3_5_language_tensor_profiles};

use crate::common::qwen3_5_moe::certified_ornith_config_bytes;

#[test]
fn should_profile_a_log_as_stored_model_float_when_decay_math_uses_float32() {
    let mut config_document =
        serde_json::from_slice::<serde_json::Value>(&certified_ornith_config_bytes())
            .expect("the certified config should parse as JSON");
    config_document["text_config"]["mamba_ssm_dtype"] = serde_json::json!("float32");
    let config_bytes =
        serde_json::to_vec(&config_document).expect("the float32 state config should serialize");
    let config = Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the float32 mamba state dtype should be accepted");

    let a_log_tensor_profile = qwen3_5_language_tensor_profiles(&config)
        .into_iter()
        .find(|tensor_profile| {
            tensor_profile.name == "language_model.model.layers.0.linear_attn.A_log"
        })
        .expect("the linear attention decay tensor should be profiled");

    assert_eq!(a_log_tensor_profile.dtype, TensorDtype::ModelFloat);
    assert_eq!(a_log_tensor_profile.shape, [32]);
}
