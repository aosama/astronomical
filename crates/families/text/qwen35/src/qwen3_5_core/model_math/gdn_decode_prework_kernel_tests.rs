//! Unit contracts for the pure pieces of the fused decode prework launch: the
//! output shapes, the launch grid, and the short-return error. The launch
//! itself and the unpacking of real outputs need live GPU work and are covered
//! by the direct-MLX numerics contract.

use super::{
    KEPT_STATE_ROW_COUNT, LANE_COUNT, PreworkShape, prework_launch_grid, prework_output_shapes,
    unpack_prework_outputs,
};
use runtime_integration::MlxRuntimeError;

const KEY_HEAD_COUNT: i32 = 16;
const VALUE_HEAD_COUNT: i32 = 32;
const HEAD_DIMENSION: i32 = 128;
const CONVOLUTION_DIMENSION: i32 =
    2 * KEY_HEAD_COUNT * HEAD_DIMENSION + VALUE_HEAD_COUNT * HEAD_DIMENSION;

fn decode_step_shape(token_count: i32) -> PreworkShape {
    PreworkShape {
        batch_size: 1,
        token_count,
        convolution_dimension: CONVOLUTION_DIMENSION,
    }
}

#[test]
fn should_shape_queries_and_keys_by_key_heads_and_values_by_value_heads() {
    let [
        queries_shape,
        keys_shape,
        values_shape,
        _next_convolution_state_shape,
    ] = prework_output_shapes(
        &decode_step_shape(4),
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
    );

    assert_eq!(queries_shape, [1, 4, KEY_HEAD_COUNT, HEAD_DIMENSION]);
    assert_eq!(keys_shape, [1, 4, KEY_HEAD_COUNT, HEAD_DIMENSION]);
    assert_eq!(values_shape, [1, 4, VALUE_HEAD_COUNT, HEAD_DIMENSION]);
}

#[test]
fn should_keep_the_rolling_rows_in_the_next_convolution_state_whatever_the_token_count() {
    for token_count in [1, KEPT_STATE_ROW_COUNT + 1] {
        let [_queries, _keys, _values, next_convolution_state_shape] = prework_output_shapes(
            &decode_step_shape(token_count),
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
            HEAD_DIMENSION,
        );

        assert_eq!(
            next_convolution_state_shape,
            [1, KEPT_STATE_ROW_COUNT, CONVOLUTION_DIMENSION],
            "state shape for {token_count} token rows"
        );
    }
}

#[test]
fn should_launch_one_threadgroup_per_token_row_and_logical_head() {
    let launch_grid = prework_launch_grid(&decode_step_shape(4), KEY_HEAD_COUNT, VALUE_HEAD_COUNT);

    assert_eq!(
        launch_grid,
        [LANE_COUNT, 4, 2 * KEY_HEAD_COUNT + VALUE_HEAD_COUNT]
    );
}

#[test]
fn should_launch_a_single_row_for_a_one_token_decode_step() {
    let launch_grid = prework_launch_grid(&decode_step_shape(1), KEY_HEAD_COUNT, VALUE_HEAD_COUNT);

    assert_eq!(launch_grid[1], 1);
}

#[test]
fn should_name_the_queries_as_the_missing_output_when_the_kernel_returned_nothing() {
    let Err(unpack_error) = unpack_prework_outputs(Vec::new()) else {
        panic!("an empty kernel return must not unpack");
    };

    match unpack_error {
        MlxRuntimeError::RuntimeOperation { description, .. } => {
            assert_eq!(
                description,
                "the fused prework kernel did not return normalized queries"
            );
        }
        other_error => panic!("expected a runtime operation error, got {other_error:?}"),
    }
}
