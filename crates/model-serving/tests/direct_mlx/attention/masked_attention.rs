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
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::SlidingWindowMaskConstruction)
            .is_some()
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
