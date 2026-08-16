use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, compute_default_rope_frequency_denominators,
    compute_yarn_rope_frequency_denominators,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_match_generated_rope_with_default_custom_denominators() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let input_values = (0..16).map(|index| index as f32 * 0.1).collect::<Vec<_>>();
    let input = runtime
        .array_from_f32(&input_values, &[1, 2, 2, 4])
        .expect("rotary input");
    let generated = runtime
        .rope(&input, 4, 10_000.0, 3)
        .expect("generated RoPE");
    let denominators =
        compute_default_rope_frequency_denominators(10_000.0, 4).expect("default denominators");
    let denominator_array = runtime
        .array_from_f32(&denominators, &[2])
        .expect("denominators");
    let mut attribution = PerformanceAttribution::enabled();
    let custom = attribution
        .measure_operation(PerformanceOperation::RotaryEmbeddingApplication, |_| {
            runtime.rope_with_custom_frequencies(&input, 4, &denominator_array, 1.0, 3)
        })
        .expect("custom RoPE");
    for (generated_value, custom_value) in generated
        .to_vec_f32()
        .expect("generated values")
        .iter()
        .zip(custom.to_vec_f32().expect("custom values"))
    {
        assert!((generated_value - custom_value).abs() < 1e-4);
    }
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::RotaryEmbeddingApplication)
            .is_some()
    );
}

#[tokio::test]
async fn should_apply_yarn_attention_factor_to_the_rotary_prefix() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let input = runtime
        .array_from_f32(&[0.1; 16], &[1, 1, 2, 8])
        .expect("input");
    let yarn = compute_yarn_rope_frequency_denominators(500_000.0, 4, 8_192, 32.0, 32.0, 1.0)
        .expect("YaRN denominators");
    let denominators = runtime
        .array_from_f32(yarn.frequency_denominators(), &[2])
        .expect("denominator array");
    let scaled = runtime
        .multiply_scalar(&input, 1.346_573_6)
        .expect("attention factor");
    let rotary_prefix = runtime
        .slice(&scaled, &[0, 0, 0, 0], &[1, 1, 2, 4], &[1, 1, 1, 1])
        .expect("prefix");
    let untouched_tail = runtime
        .slice(&input, &[0, 0, 0, 4], &[1, 1, 2, 8], &[1, 1, 1, 1])
        .expect("tail");
    let prepared = runtime
        .concatenate_axis(&[&rotary_prefix, &untouched_tail], 3)
        .expect("prepared input");
    let rotated = runtime
        .rope_with_custom_frequencies(&prepared, 4, &denominators, 1.0, 0)
        .expect("YaRN RoPE");
    assert_eq!(rotated.shape(), vec![1, 1, 2, 8]);
}

#[tokio::test]
async fn should_compute_softplus_gate_in_float32_then_restore_activation_dtype() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let output = runtime
        .astype(
            &runtime
                .array_from_f32(&[1.0, 2.0], &[1, 1, 2])
                .expect("output"),
            MlxDtype::BFloat16,
        )
        .expect("BF16 output");
    let logits = runtime
        .astype(
            &runtime.array_from_f32(&[0.0], &[1, 1, 1]).expect("logits"),
            MlxDtype::BFloat16,
        )
        .expect("BF16 logits");
    let gated = runtime
        .apply_softplus_attention_gate(&output, &logits)
        .expect("softplus gate");
    assert_eq!(gated.dtype(), MlxDtype::BFloat16);
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits"),
    )
    .expect("direct MLX runtime")
}
