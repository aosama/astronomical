use astronomical_mlx_c_rust::MlxArray;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes;

use super::{apply_declared_jangtq2_rms_norm_convention, is_language_rms_norm_weight_tensor};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the jangtq2 norm test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn config_with_norm_convention(norm_convention: &str) -> Qwen3_5Config {
    let mut config_value =
        serde_json::from_slice::<serde_json::Value>(&frozen_ornith_1_0_config_bytes())
            .expect("the frozen Qwen3.5 config should decode");
    config_value["norm_convention"] = serde_json::Value::String(norm_convention.to_owned());
    let config_bytes = serde_json::to_vec(&config_value)
        .expect("the synthetic norm-convention config should serialize");
    Qwen3_5Config::from_json_bytes(&config_bytes)
        .expect("the synthetic norm-convention config should parse")
}

fn test_norm_weight(runtime: &MlxRuntime) -> MlxArray {
    runtime
        .array_from_f32(&[0.25, -0.5], &[2])
        .expect("the synthetic norm weight should be valid")
}

#[test]
#[serial]
fn should_add_one_to_declared_jangtq2_language_rms_norm_weights_at_binding() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("qwen3_5_language_mlx_plus_one");

    let adjusted_norm_weight = apply_declared_jangtq2_rms_norm_convention(
        &runtime,
        &qwen3_5_config,
        "language_model.model.layers.0.input_layernorm.weight",
        test_norm_weight(&runtime),
    )
    .expect("the JANGH2 RMSNorm deviation should receive its unit scale");
    let adjusted_norm_values = adjusted_norm_weight
        .to_vec_f32()
        .expect("the adjusted RMSNorm weight should be readable");

    assert_eq!(adjusted_norm_values, [1.25, 0.5]);
    assert!(is_language_rms_norm_weight_tensor(
        "language_model.model.layers.0.self_attn.q_norm.weight"
    ));
    assert!(is_language_rms_norm_weight_tensor(
        "language_model.model.layers.0.linear_attn.norm.weight"
    ));
    assert!(!is_language_rms_norm_weight_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    ));
}

#[test]
#[serial]
fn should_preserve_direct_rms_norm_weights_when_the_artifact_has_no_plus_one_convention() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("direct");

    let direct_norm_weight = apply_declared_jangtq2_rms_norm_convention(
        &runtime,
        &qwen3_5_config,
        "language_model.model.layers.0.input_layernorm.weight",
        test_norm_weight(&runtime),
    )
    .expect("the direct RMSNorm convention should retain its stored weight");

    assert_eq!(
        direct_norm_weight
            .to_vec_f32()
            .expect("the direct RMSNorm weight should be readable"),
        [0.25, -0.5]
    );
}
