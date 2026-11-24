//! The plain-operation attention a training differentiates, against MLX's
//! fused attention serving runs, on grouped heads.

use serial_test::serial;

use super::*;
use model_primitives::synthetic_weights_test_support::{
    initialize_test_runtime, shakespeare_values,
};
use runtime_integration::MlxRuntime;

const QUERY_HEAD_COUNT: i32 = 4;
const KEY_VALUE_HEAD_COUNT: i32 = 2;
const HEAD_DIMENSION: i32 = 16;
const KEY_TOKEN_COUNT: i32 = 9;
/// Both routes compute in float32 and add in different orders.
const ROUNDING_TOLERANCE: f32 = 1e-5;
/// One TF32 rounding of values about 0.1 in size.
const TF32_ROUNDING_TOLERANCE: f32 = 1e-4;

/// Reduced on the device: a sliced view read straight to the host would be
/// read in its parent's layout.
fn largest_gap(runtime: &MlxRuntime, left: &MlxArray, right: &MlxArray) -> f32 {
    let gaps = runtime
        .abs(&runtime.subtract(left, right).unwrap())
        .unwrap();
    runtime.max(&gaps, false).unwrap().to_vec_f32().unwrap()[0]
}

fn keys_and_values(runtime: &MlxRuntime) -> (MlxArray, MlxArray) {
    let key_shape = [1, KEY_VALUE_HEAD_COUNT, KEY_TOKEN_COUNT, HEAD_DIMENSION];
    (
        shakespeare_values(runtime, &key_shape, 500, 0.0),
        shakespeare_values(runtime, &key_shape, 900, 0.0),
    )
}

#[test]
#[serial]
fn should_match_fused_causal_attention_for_a_run_of_queries_after_a_prefix() {
    let runtime = initialize_test_runtime();
    let (keys, values) = keys_and_values(&runtime);
    // Five queries are the last five positions; four earlier keys are the prefix.
    let queries = shakespeare_values(&runtime, &[1, QUERY_HEAD_COUNT, 5, HEAD_DIMENSION], 0, 0.0);

    let fused = runtime
        .causal_scaled_dot_product_attention(&queries, &keys, &values, 0.25)
        .unwrap();
    let plain = plain_operation_attention(&runtime, &queries, &keys, &values, 0.25).unwrap();

    assert_eq!(fused.shape(), plain.shape());
    assert!(largest_gap(&runtime, &fused, &plain) <= ROUNDING_TOLERANCE);
}

#[test]
#[serial]
fn should_match_fused_attention_for_one_query_seeing_every_key() {
    let runtime = initialize_test_runtime();
    let (keys, values) = keys_and_values(&runtime);
    let queries = shakespeare_values(&runtime, &[1, QUERY_HEAD_COUNT, 1, HEAD_DIMENSION], 0, 0.0);

    let fused = runtime
        .scaled_dot_product_attention(&queries, &keys, &values, 0.25)
        .unwrap();
    let plain = plain_operation_attention(&runtime, &queries, &keys, &values, 0.25).unwrap();

    assert!(largest_gap(&runtime, &fused, &plain) <= ROUNDING_TOLERANCE);
}

#[test]
#[serial]
fn should_hide_later_keys_from_an_earlier_query() {
    let runtime = initialize_test_runtime();
    let (keys, values) = keys_and_values(&runtime);
    let prefix_keys = runtime
        .slice(
            &keys,
            &[0, 0, 0, 0],
            &[1, KEY_VALUE_HEAD_COUNT, 1, HEAD_DIMENSION],
            &[1, 1, 1, 1],
        )
        .unwrap();
    let prefix_values = runtime
        .slice(
            &values,
            &[0, 0, 0, 0],
            &[1, KEY_VALUE_HEAD_COUNT, 1, HEAD_DIMENSION],
            &[1, 1, 1, 1],
        )
        .unwrap();

    // A run of all nine queries: the first sees only the first key and value,
    // so its output is exactly that value.
    let all_queries = shakespeare_values(
        &runtime,
        &[1, QUERY_HEAD_COUNT, KEY_TOKEN_COUNT, HEAD_DIMENSION],
        0,
        0.0,
    );
    let first_query = runtime
        .slice(
            &all_queries,
            &[0, 0, 0, 0],
            &[1, QUERY_HEAD_COUNT, 1, HEAD_DIMENSION],
            &[1, 1, 1, 1],
        )
        .unwrap();
    let plain = plain_operation_attention(&runtime, &all_queries, &keys, &values, 0.25).unwrap();
    let first_row = runtime
        .slice(
            &plain,
            &[0, 0, 0, 0],
            &[1, QUERY_HEAD_COUNT, 1, HEAD_DIMENSION],
            &[1, 1, 1, 1],
        )
        .unwrap();
    let alone =
        plain_operation_attention(&runtime, &first_query, &prefix_keys, &prefix_values, 0.25)
            .unwrap();

    // Nine-row and one-row products of the weights and values round
    // differently in MLX's float32 (TF32) matrix product on this hardware,
    // about 1e-4 of the values' size; a leaked later key would move the row
    // by the values' own size, about 0.1.
    assert!(largest_gap(&runtime, &first_row, &alone) <= TF32_ROUNDING_TOLERANCE);
}
