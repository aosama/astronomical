use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, build_causal_sliding_window_mask,
    sliding_window_visibility_table,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

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
        .astype(&mask, astronomical_runtime_integration::MlxDtype::Float32)
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
