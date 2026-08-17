use astronomical_model_serving::{PerformanceAttribution, build_causal_sliding_window_mask};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

/// One layer's descriptor-derived grouped-query attention geometry.
#[derive(Clone, Copy)]
pub(crate) struct AttentionGeometry {
    pub(crate) row_name: &'static str,
    pub(crate) query_head_count: i32,
    pub(crate) key_value_head_count: i32,
    pub(crate) query_token_count: i32,
    pub(crate) prefix_token_count: i32,
    pub(crate) head_width: i32,
    pub(crate) activation_dtype: MlxDtype,
    pub(crate) visibility: AttentionVisibility,
}

/// The caller owns whether a layer uses global causal or sliding visibility.
#[derive(Clone, Copy)]
pub(crate) enum AttentionVisibility {
    Full,
    Sliding { window_size: i32 },
}

/// Direct MLX is the widest reliable boundary for the neutral primitive. This
/// matrix models all-full, all-sliding, and mixed callers without importing a
/// concrete model family's geometry into neutral primitive coverage.
#[tokio::test]
async fn should_match_operations_reference_for_full_sliding_and_mixed_schedules() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let all_full_schedule = [AttentionGeometry {
        row_name: "generic_all_full",
        query_head_count: 6,
        key_value_head_count: 3,
        query_token_count: 3,
        prefix_token_count: 2,
        head_width: 4,
        activation_dtype: MlxDtype::Float32,
        visibility: AttentionVisibility::Full,
    }];
    let all_sliding_schedule = [AttentionGeometry {
        row_name: "generic_all_sliding",
        query_head_count: 12,
        key_value_head_count: 4,
        query_token_count: 2,
        prefix_token_count: 5,
        head_width: 8,
        activation_dtype: MlxDtype::BFloat16,
        visibility: AttentionVisibility::Sliding { window_size: 4 },
    }];
    let mixed_schedule = [
        AttentionGeometry {
            row_name: "generic_mixed_full",
            query_head_count: 8,
            key_value_head_count: 2,
            query_token_count: 2,
            prefix_token_count: 3,
            head_width: 16,
            activation_dtype: MlxDtype::BFloat16,
            visibility: AttentionVisibility::Full,
        },
        AttentionGeometry {
            row_name: "generic_mixed_sliding",
            query_head_count: 10,
            key_value_head_count: 2,
            query_token_count: 3,
            prefix_token_count: 5,
            head_width: 12,
            activation_dtype: MlxDtype::BFloat16,
            visibility: AttentionVisibility::Sliding { window_size: 6 },
        },
    ];

    for schedule in [
        &all_full_schedule[..],
        &all_sliding_schedule,
        &mixed_schedule,
    ] {
        for geometry in schedule {
            assert_attention_matches_operations_reference(&runtime, *geometry);
        }
    }
}

pub(crate) fn assert_attention_matches_operations_reference(
    runtime: &MlxRuntime,
    geometry: AttentionGeometry,
) {
    let key_token_count = geometry.prefix_token_count + geometry.query_token_count;
    let queries = deterministic_array(
        runtime,
        &[
            1,
            geometry.query_head_count,
            geometry.query_token_count,
            geometry.head_width,
        ],
        geometry.activation_dtype,
        17,
        32.0,
    );
    let keys = deterministic_array(
        runtime,
        &[
            1,
            geometry.key_value_head_count,
            key_token_count,
            geometry.head_width,
        ],
        geometry.activation_dtype,
        13,
        40.0,
    );
    let values = deterministic_array(
        runtime,
        &[
            1,
            geometry.key_value_head_count,
            key_token_count,
            geometry.head_width,
        ],
        geometry.activation_dtype,
        19,
        24.0,
    );
    let mask_window_size = match geometry.visibility {
        AttentionVisibility::Full => geometry.prefix_token_count + geometry.query_token_count + 1,
        AttentionVisibility::Sliding { window_size } => window_size,
    };
    let mask = build_causal_sliding_window_mask(
        runtime,
        geometry.prefix_token_count,
        geometry.query_token_count,
        0,
        key_token_count,
        mask_window_size,
        &mut PerformanceAttribution::disabled(),
    )
    .unwrap_or_else(|error| panic!("{} mask should build: {error}", geometry.row_name));
    let attention_scale = 1.0 / (geometry.head_width as f32).sqrt();
    let fused_output = runtime
        .masked_scaled_dot_product_attention(&queries, &keys, &values, attention_scale, &mask)
        .unwrap_or_else(|error| {
            panic!(
                "{} fused attention should build: {error}",
                geometry.row_name
            )
        });
    assert_eq!(
        fused_output.dtype(),
        geometry.activation_dtype,
        "{}",
        geometry.row_name
    );

    // Evaluate the exact low-precision inputs before the host reference so the
    // comparison isolates attention arithmetic rather than source conversion.
    let query_values = float32_values(runtime, &queries, geometry.row_name);
    let key_values = float32_values(runtime, &keys, geometry.row_name);
    let value_values = float32_values(runtime, &values, geometry.row_name);
    let expected_output = attention_operations_reference(
        &query_values,
        &key_values,
        &value_values,
        geometry,
        key_token_count,
        attention_scale,
    );
    let actual_output = float32_values(runtime, &fused_output, geometry.row_name);
    // The fused MLX kernel and the decomposed host reference can associate the
    // dot product differently while preserving the same Float32 contract.
    let tolerance = if geometry.activation_dtype == MlxDtype::Float32 {
        2e-4
    } else {
        2e-2
    };
    assert_values_close(
        geometry.row_name,
        &actual_output,
        &expected_output,
        tolerance,
    );
}

fn attention_operations_reference(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    geometry: AttentionGeometry,
    key_token_count: i32,
    attention_scale: f32,
) -> Vec<f32> {
    let query_head_count = geometry.query_head_count as usize;
    let key_value_head_count = geometry.key_value_head_count as usize;
    let query_token_count = geometry.query_token_count as usize;
    let key_token_count = key_token_count as usize;
    let head_width = geometry.head_width as usize;
    let query_heads_per_key_value_head = query_head_count / key_value_head_count;
    let mut output = vec![0.0_f32; query_head_count * query_token_count * head_width];

    for query_head_index in 0..query_head_count {
        let key_value_head_index = query_head_index / query_heads_per_key_value_head;
        for query_token_index in 0..query_token_count {
            let query_absolute_position = geometry.prefix_token_count as usize + query_token_index;
            let mut scores = vec![f32::NEG_INFINITY; key_token_count];
            for key_token_index in 0..key_token_count {
                if token_is_visible(
                    geometry.visibility,
                    query_absolute_position,
                    key_token_index,
                ) {
                    let mut dot_product = 0.0_f32;
                    for head_dimension_index in 0..head_width {
                        dot_product += queries[array_index(
                            query_head_index,
                            query_token_index,
                            head_dimension_index,
                            query_token_count,
                            head_width,
                        )] * keys[array_index(
                            key_value_head_index,
                            key_token_index,
                            head_dimension_index,
                            key_token_count,
                            head_width,
                        )];
                    }
                    scores[key_token_index] = attention_scale * dot_product;
                }
            }
            let maximum_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut score_exponentials = scores
                .iter()
                .map(|score| {
                    if score.is_finite() {
                        (*score - maximum_score).exp()
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let exponential_sum = score_exponentials.iter().sum::<f32>();
            for score_exponential in &mut score_exponentials {
                *score_exponential /= exponential_sum;
            }
            for head_dimension_index in 0..head_width {
                let mut weighted_value = 0.0_f32;
                for key_token_index in 0..key_token_count {
                    weighted_value += score_exponentials[key_token_index]
                        * values[array_index(
                            key_value_head_index,
                            key_token_index,
                            head_dimension_index,
                            key_token_count,
                            head_width,
                        )];
                }
                output[array_index(
                    query_head_index,
                    query_token_index,
                    head_dimension_index,
                    query_token_count,
                    head_width,
                )] = weighted_value;
            }
        }
    }
    output
}

fn token_is_visible(
    visibility: AttentionVisibility,
    query_absolute_position: usize,
    key_absolute_position: usize,
) -> bool {
    if key_absolute_position > query_absolute_position {
        return false;
    }
    match visibility {
        AttentionVisibility::Full => true,
        AttentionVisibility::Sliding { window_size } => {
            query_absolute_position < key_absolute_position + window_size as usize
        }
    }
}

fn array_index(
    head_index: usize,
    token_index: usize,
    dimension_index: usize,
    token_count: usize,
    head_width: usize,
) -> usize {
    (head_index * token_count + token_index) * head_width + dimension_index
}

fn deterministic_array(
    runtime: &MlxRuntime,
    shape: &[i32],
    dtype: MlxDtype,
    period: usize,
    divisor: f32,
) -> MlxArray {
    let element_count = shape.iter().product::<i32>() as usize;
    let midpoint = (period / 2) as f32;
    let values = (0..element_count)
        .map(|element_index| ((element_index % period) as f32 - midpoint) / divisor)
        .collect::<Vec<_>>();
    runtime
        .array_from_f32(&values, shape)
        .and_then(|array| runtime.astype(&array, dtype))
        .expect("deterministic attention input should be valid")
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray, row_name: &str) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| runtime.build_contiguous_row_major_copy(&float32_array))
        .and_then(|contiguous_array| contiguous_array.to_vec_f32())
        .unwrap_or_else(|error| panic!("{row_name} should evaluate: {error}"))
}

fn assert_values_close(row_name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{row_name}");
    for (element_index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= tolerance * comparison_scale,
            "{row_name} element {element_index}: expected {expected_value}, got {actual_value}"
        );
    }
}

pub(crate) fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("grouped-query test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
