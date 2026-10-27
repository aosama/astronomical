use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_c_rust::MlxArray;
use serial_test::serial;

use crate::qwen3_5_core::configuration::Qwen3_5Config;
use crate::qwen3_5_core::frozen_config_test_support::frozen_ornith_1_0_config_bytes;

use super::{
    apply_rms_norm_storage_convention, is_language_rms_norm_weight_tensor,
    is_published_pytorch_convolution_layout, is_raw_hugging_face_norm_offset_weight_tensor,
};

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

    let adjusted_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        false,
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

    let direct_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        false,
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

#[test]
fn should_classify_raw_hugging_face_norm_offset_tensor_names() {
    assert!(is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.norm.weight"
    ));
    assert!(is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.0.input_layernorm.weight"
    ));
    assert!(is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.3.post_attention_layernorm.weight"
    ));
    assert!(is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.3.self_attn.q_norm.weight"
    ));
    assert!(is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.3.self_attn.k_norm.weight"
    ));
}

#[test]
fn should_exclude_non_offset_tensors_from_the_raw_hugging_face_convention() {
    // The gated linear-attention output norm keeps its direct gamma in every
    // checkpoint convention; offsetting it would corrupt every linear layer.
    assert!(!is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.0.linear_attn.norm.weight"
    ));
    assert!(!is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.embed_tokens.weight"
    ));
    assert!(!is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.lm_head.weight"
    ));
    assert!(!is_raw_hugging_face_norm_offset_weight_tensor(
        "language_model.model.layers.0.mlp.up_proj.weight"
    ));
}

#[test]
fn should_detect_the_published_pytorch_convolution_layout_exactly() {
    assert!(is_published_pytorch_convolution_layout(&[8, 1, 3], 8, 3));
    assert!(!is_published_pytorch_convolution_layout(&[8, 3, 1], 8, 3));
    assert!(!is_published_pytorch_convolution_layout(&[8, 2, 3], 8, 3));
    assert!(!is_published_pytorch_convolution_layout(&[8, 1], 8, 3));
}

#[test]
#[serial]
fn should_add_the_unit_scale_to_raw_hugging_face_norm_weights() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("direct");

    let adjusted_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        true,
        "language_model.model.layers.0.input_layernorm.weight",
        test_norm_weight(&runtime),
    )
    .expect("the raw Hugging Face offset norm should receive its unit scale");

    assert_eq!(
        adjusted_norm_weight
            .to_vec_f32()
            .expect("the offset-adjusted RMSNorm weight should be readable"),
        [1.25, 0.5]
    );
}

#[test]
#[serial]
fn should_pass_norms_through_when_the_convolution_is_already_mlx_converted() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("direct");

    let unchanged_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        false,
        "language_model.model.layers.0.input_layernorm.weight",
        test_norm_weight(&runtime),
    )
    .expect("an MLX-converted checkpoint should keep its stored gamma");

    assert_eq!(
        unchanged_norm_weight
            .to_vec_f32()
            .expect("the stored RMSNorm weight should be readable"),
        [0.25, -0.5]
    );
}

#[test]
#[serial]
fn should_leave_the_gated_linear_attention_norm_direct_under_the_raw_hugging_face_convention() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("direct");

    let direct_gated_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        true,
        "language_model.model.layers.0.linear_attn.norm.weight",
        test_norm_weight(&runtime),
    )
    .expect("the gated linear-attention norm should keep its direct gamma");

    assert_eq!(
        direct_gated_norm_weight
            .to_vec_f32()
            .expect("the gated linear-attention norm should be readable"),
        [0.25, -0.5]
    );
}

#[test]
#[serial]
fn should_scale_only_once_when_both_conventions_would_match_the_same_tensor() {
    let runtime = test_runtime();
    let qwen3_5_config = config_with_norm_convention("qwen3_5_language_mlx_plus_one");

    let adjusted_norm_weight = apply_rms_norm_storage_convention(
        &runtime,
        &qwen3_5_config,
        true,
        "language_model.model.layers.0.input_layernorm.weight",
        test_norm_weight(&runtime),
    )
    .expect("the overlapping conventions should scale the tensor exactly once");

    assert_eq!(
        adjusted_norm_weight
            .to_vec_f32()
            .expect("the convention-adjusted RMSNorm weight should be readable"),
        [1.25, 0.5]
    );
}
