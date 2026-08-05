use astronomical_model_serving::{
    Qwen3_5Config, TensorDtype, qwen3_5_mtp_tensor_profiles, qwen3_5_quantized_mtp_tensor_names,
};

use crate::common::qwen3_5_moe::certified_optiq_ornith_config_bytes;

#[test]
fn should_build_the_complete_one_layer_qwen_quantized_mtp_tensor_namespace() {
    let optiq_config = Qwen3_5Config::from_json_bytes(&certified_optiq_ornith_config_bytes())
        .expect("the OptiQ configuration should parse");
    let expected_tensor_names = qwen3_5_quantized_mtp_tensor_names(&optiq_config);

    assert_eq!(expected_tensor_names.len(), 42);
    assert!(expected_tensor_names.contains("language_model.mtp.fc.weight"));
    assert!(expected_tensor_names.contains("language_model.mtp.pre_fc_norm_embedding.weight"));
    assert!(expected_tensor_names.contains("language_model.mtp.layers.0.self_attn.q_proj.scales"));
    assert!(
        expected_tensor_names
            .contains("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.biases")
    );
}

#[test]
fn should_describe_the_oq4e_mtp_fusion_and_expert_tensor_shapes() {
    let optiq_config = Qwen3_5Config::from_json_bytes(&certified_optiq_ornith_config_bytes())
        .expect("the OptiQ configuration should parse");
    let mtp_tensor_profiles = qwen3_5_mtp_tensor_profiles(&optiq_config);

    assert_eq!(mtp_tensor_profiles.len(), 42);
    assert!(mtp_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name == "language_model.mtp.fc.weight"
            && tensor_profile.dtype == TensorDtype::ModelFloat
            && tensor_profile.shape == [2_048, 4_096]
    }));
    assert!(mtp_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name == "language_model.mtp.layers.0.self_attn.q_proj.weight"
            && tensor_profile.dtype == TensorDtype::UInt32
            && tensor_profile.shape
                == [
                    8_192,
                    optiq_config.hidden_size() as usize
                        * optiq_config
                            .quantization_profile_for_module(
                                "language_model.mtp.layers.0.self_attn.q_proj",
                            )
                            .bits as usize
                        / 32,
                ]
    }));
    assert!(mtp_tensor_profiles.iter().any(|tensor_profile| {
        tensor_profile.name == "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"
            && tensor_profile.dtype == TensorDtype::UInt32
            && tensor_profile.shape == [256, 512, 256]
    }));
}
