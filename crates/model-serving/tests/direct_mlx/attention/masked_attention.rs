use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, build_causal_sliding_window_mask,
    sliding_window_visibility_table,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

#[tokio::test]
async fn should_match_fused_causal_attention_with_an_array_causal_mask() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let queries = runtime
        .array_from_f32(&[1.0, 1.0], &[1, 1, 2, 1])
        .expect("queries");
    let keys = runtime
        .array_from_f32(&[0.0, 0.0], &[1, 1, 2, 1])
        .expect("keys");
    let values = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2, 1])
        .expect("values");
    let fused = runtime
        .causal_scaled_dot_product_attention(&queries, &keys, &values, 1.0)
        .expect("fused causal attention should succeed");
    let mut attribution = PerformanceAttribution::enabled();
    let mask = build_causal_sliding_window_mask(&runtime, 0, 2, 0, 2, 8, &mut attribution)
        .expect("a large window should produce a causal mask");
    let masked = runtime
        .masked_scaled_dot_product_attention(&queries, &keys, &values, 1.0, &mask)
        .expect("array-masked attention should succeed");
    assert_eq!(
        fused.to_vec_f32().expect("fused output should evaluate"),
        masked.to_vec_f32().expect("masked output should evaluate")
    );
    let mask_measurement = attribution
        .operation_measurement(PerformanceOperation::SlidingWindowMaskConstruction)
        .expect("enabled attribution should retain the mask operation boundaries");
    assert_eq!(mask_measurement.occurrence_count(), 1);
    assert!(
        mask_measurement.last_ended_offset_nanoseconds()
            >= mask_measurement.first_started_offset_nanoseconds()
    );
}

#[tokio::test]
async fn should_match_the_cpu_visibility_table_for_a_prefix_plus_chunk() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut attribution = PerformanceAttribution::disabled();
    let mask = build_causal_sliding_window_mask(&runtime, 6, 4, 0, 10, 4, &mut attribution)
        .expect("the mask should build");
    let actual = runtime
        .astype(&mask, astronomical_mlx_c_rust::MlxDtype::Float32)
        .expect("the mask should cast")
        .to_vec_f32()
        .expect("the mask should evaluate");
    let expected = sliding_window_visibility_table(6, 4, 0, 10, 4)
        .expect("the CPU contract should build")
        .into_iter()
        .flatten()
        .map(|is_visible| if is_visible { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::SlidingWindowMaskConstruction)
            .is_none()
    );
}

#[tokio::test]
async fn should_reject_negative_or_overflowing_absolute_mask_geometry() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    for (first_query_position, query_tokens, first_key_position, key_tokens, window_size) in [
        (-1, 1, 0, 1, 4),
        (0, 1, -1, 1, 4),
        (i32::MAX, 1, 0, 1, 4),
        (0, 1, i32::MAX - 1, 1, 4),
    ] {
        build_causal_sliding_window_mask(
            &runtime,
            first_query_position,
            query_tokens,
            first_key_position,
            key_tokens,
            window_size,
            &mut PerformanceAttribution::disabled(),
        )
        .expect_err("invalid absolute mask geometry must fail before MLX execution");
    }
}

#[tokio::test]
async fn should_reject_zero_head_attention_without_panicking() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let queries = runtime
        .array_from_f32(&[], &[1, 0, 1, 4])
        .expect("zero-head queries should be representable");
    let keys = runtime
        .array_from_f32(&[], &[1, 0, 1, 4])
        .expect("zero-head keys should be representable");
    let values = runtime
        .array_from_f32(&[], &[1, 0, 1, 4])
        .expect("zero-head values should be representable");

    runtime
        .scaled_dot_product_attention(&queries, &keys, &values, 0.5)
        .expect_err("zero attention heads must return a typed error before modulo validation");
}

#[tokio::test]
async fn should_compile_nax_dsplit_attention_kernels_for_wide_head_dimensions() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let queries_256 = wide_bfloat16_attention_input(&runtime, 2, 1024, 256);
    let keys_256 = wide_bfloat16_attention_input(&runtime, 2, 2048, 256);
    let values_256 = wide_bfloat16_attention_input(&runtime, 2, 2048, 256);

    let causal_output = runtime
        .causal_scaled_dot_product_attention(&queries_256, &keys_256, &values_256, 1.0)
        .unwrap_or_else(|error| {
            panic!("causal wide attention should JIT-compile for head dimension 256: {error}")
        });
    assert_wide_attention_output_is_finite(&runtime, &causal_output, 2 * 1024 * 256);

    let mut attribution = PerformanceAttribution::disabled();
    let causal_mask =
        build_causal_sliding_window_mask(&runtime, 0, 1024, 0, 2048, 2048, &mut attribution)
            .expect("a full-history window should produce a causal mask");
    let bfloat16_causal_mask = runtime
        .astype(&causal_mask, MlxDtype::BFloat16)
        .expect("the causal mask should cast to bfloat16");
    let masked_output = runtime
        .masked_scaled_dot_product_attention(
            &queries_256,
            &keys_256,
            &values_256,
            1.0,
            &bfloat16_causal_mask,
        )
        .unwrap_or_else(|error| {
            panic!("masked wide attention should JIT-compile for head dimension 256: {error}")
        });
    assert_wide_attention_output_is_finite(&runtime, &masked_output, 2 * 1024 * 256);

    // The head-dimension 512 fused kernel additionally needs at least 1024
    // query blocks (batch * heads * ceil(query_length / 32)) before dispatch
    // leaves the unfused fallback, so this case uses 32 heads.
    let queries_512 = wide_bfloat16_attention_input(&runtime, 32, 1024, 512);
    let keys_512 = wide_bfloat16_attention_input(&runtime, 32, 1024, 512);
    let values_512 = wide_bfloat16_attention_input(&runtime, 32, 1024, 512);
    let wide_output = runtime
        .causal_scaled_dot_product_attention(&queries_512, &keys_512, &values_512, 1.0)
        .unwrap_or_else(|error| {
            panic!("causal wide attention should JIT-compile for head dimension 512: {error}")
        });
    assert_wide_attention_output_is_finite(&runtime, &wide_output, 32 * 1024 * 512);
}

// Bfloat16 inputs and query lengths of at least 1024 tokens are load-bearing:
// float32 queries skip the NAX dispatch gate and shorter queries stay on the
// unfused fallback, so only this shape family reaches the head-split (dsplit)
// Metal JIT path this regression test exists to protect.
fn wide_bfloat16_attention_input(
    runtime: &MlxRuntime,
    head_count: usize,
    token_count: usize,
    head_dimension: usize,
) -> MlxArray {
    let element_count = head_count * token_count * head_dimension;
    let element_values: Vec<f32> = (0..element_count)
        .map(|element_index| ((element_index % 13) as f32 - 6.0) * 0.125)
        .collect();
    let float_input = runtime
        .array_from_f32(
            &element_values,
            &[
                1,
                head_count as i32,
                token_count as i32,
                head_dimension as i32,
            ],
        )
        .expect("wide attention input should build from deterministic values");
    let bfloat16_input = runtime
        .astype(&float_input, MlxDtype::BFloat16)
        .expect("wide attention input should cast to bfloat16");
    // Evaluating before the float source drops keeps the lazy graph from
    // holding every float32 intermediate alive during the attention eval,
    // which matters for the head-dimension 512 case under the direct-MLX
    // test memory ceiling.
    bfloat16_input
        .evaluate()
        .expect("wide attention input should evaluate");
    bfloat16_input
}

fn assert_wide_attention_output_is_finite(
    runtime: &MlxRuntime,
    attention_output: &MlxArray,
    expected_element_count: usize,
) {
    let output_f32 = runtime
        .astype(attention_output, MlxDtype::Float32)
        .expect("wide attention output should cast to float32");
    let output_values = output_f32
        .to_vec_f32()
        .expect("wide attention output should evaluate");
    assert_eq!(
        output_values.len(),
        expected_element_count,
        "wide attention should produce one output value per query position"
    );
    assert!(
        output_values.iter().all(|value| value.is_finite()),
        "wide attention output should stay finite"
    );
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
