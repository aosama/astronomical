use astronomical_model_serving::qwen3_5_moe_full_attention_step;
use astronomical_runtime_integration::{MlxCompiledElementwiseGraphs, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_apply_one_cached_grouped_query_attention_step_with_output_gating() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the attention test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the attention elementwise graphs should compile");
    let rotated_queries = runtime
        .array_from_f32(&[0.0; 4], &[1, 2, 1, 2])
        .expect("the rotated queries should be valid");
    let active_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 2, 2])
        .expect("the active keys should be valid");
    let active_values = runtime
        .array_from_f32(&[2.0, 4.0, 4.0, 6.0], &[1, 1, 2, 2])
        .expect("the active values should be valid");
    let output_gate = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4])
        .expect("the output gate should be valid");
    let gated_output = qwen3_5_moe_full_attention_step(
        &runtime,
        &compiled_elementwise_graphs,
        &rotated_queries,
        &active_keys,
        &active_values,
        &output_gate,
        2.0_f32.sqrt().recip(),
    )
    .expect("the full-attention step should build a valid graph");

    assert_eq!(gated_output.shape(), vec![1, 1, 4]);
    assert_f32_close(
        &gated_output
            .to_vec_f32()
            .expect("the gated output should evaluate as float32"),
        &[1.5, 2.5, 1.5, 2.5],
    );
}

#[tokio::test]
async fn should_apply_causal_attention_across_multiple_prefill_tokens() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the attention test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the attention elementwise graphs should compile");
    let rotated_queries = runtime
        .array_from_f32(&[0.0; 8], &[1, 2, 2, 2])
        .expect("the rotated queries should be valid");
    let active_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 2, 2])
        .expect("the active keys should be valid");
    let active_values = runtime
        .array_from_f32(&[2.0, 4.0, 4.0, 6.0], &[1, 1, 2, 2])
        .expect("the active values should be valid");
    let output_gate = runtime
        .array_from_f32(&[0.0; 8], &[1, 2, 4])
        .expect("the output gate should be valid");
    let gated_output = qwen3_5_moe_full_attention_step(
        &runtime,
        &compiled_elementwise_graphs,
        &rotated_queries,
        &active_keys,
        &active_values,
        &output_gate,
        2.0_f32.sqrt().recip(),
    )
    .expect("the causal full-attention step should build a valid graph");

    assert_eq!(gated_output.shape(), vec![1, 2, 4]);
    assert_f32_close(
        &gated_output
            .to_vec_f32()
            .expect("the gated output should evaluate as float32"),
        &[1.0, 2.0, 1.0, 2.0, 1.5, 2.5, 1.5, 2.5],
    );
}

fn assert_f32_close(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        assert!(
            (*actual_value - *expected_value).abs() <= 1e-6,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}
