//! Parity contracts for the `qwen4_exp` hyper-connection executor.
//!
//! The user-visible outcome under test: the production executor produces the
//! same mixed streams, normalized streams, and injections as the direct-MLX
//! oracle, which was itself proven against the pinned host algebra and the
//! hermetic contract's exact values. A production route that drifts — a
//! reordered gate, a dropped scale, a widened activation — fails here.
//!
//! These tests run at the direct-MLX boundary and are serial.

use astronomical_model_serving::{HyperConnectionExecutor, StreamMixingPlan};
use astronomical_runtime_integration::MlxRuntime;

use crate::direct_mlx::qwen4_exp::{
    DeterministicValues, assert_f32_close, f32_array, oracle_test_runtime,
};

fn executor_plan() -> StreamMixingPlan {
    StreamMixingPlan {
        stream_count: 2,
        stream_width: 4,
        low_rank: 3,
        rms_norm_epsilon: 1.0e-6,
    }
}

fn contract_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hyper_input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let norm_weights = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 0.25, 0.75];
    let down = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let up = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 0.0, 1.0, //
        0.0, 1.0, 1.0, //
        1.0, 1.0, 1.0,
    ];
    let inject = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    (hyper_input, norm_weights, down, up, inject)
}

#[tokio::test]
async fn should_reproduce_the_hermetic_contract_values_through_the_executor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let executor = HyperConnectionExecutor::new(executor_plan());
    let (hyper_input, norm_weights, down, up, inject) = contract_inputs();
    let input_array = f32_array(&runtime, &hyper_input, &[1, 8]).expect("input constructs");
    let norm_array = f32_array(&runtime, &norm_weights, &[8]).expect("norm constructs");
    let down_array = f32_array(&runtime, &down, &[3, 8]).expect("down constructs");
    let up_array = f32_array(&runtime, &up, &[8, 3]).expect("up constructs");
    let mut attribution = astronomical_model_serving::PerformanceAttribution::disabled();
    let (mixed, normalized) = executor
        .gated_mix(
            &runtime,
            &norm_array,
            &down_array,
            &up_array,
            &input_array,
            &mut attribution,
        )
        .expect("executor mixing should run");
    mixed.evaluate().expect("mixed should evaluate");
    normalized.evaluate().expect("normalized should evaluate");
    let mixed_values = mixed.to_vec_f32().expect("mixed copies");
    // The per-stream normalization path slices and stacks, which accumulates
    // slightly differently from the oracle's composition; 5e-5 is the
    // observed f32 noise floor for this shape and stays far inside any
    // activation-dtype meaning.
    assert_f32_close(
        &mixed_values,
        &[0.456_878_934, 0.304_467_565, 0.874_527_157, 1.137_607_931],
        5.0e-5,
        "executor mixing must reproduce the hermetic contract values",
    );
    // Combine must inject into the raw residual with the pinned weights.
    let inject_array = f32_array(&runtime, &inject, &[2, 8]).expect("inject constructs");
    let block_output = vec![0.5, -0.5, 1.0, 2.0];
    let block_array = f32_array(&runtime, &block_output, &[1, 4]).expect("block constructs");
    let combined = executor
        .gated_combine(
            &runtime,
            &inject_array,
            &block_array,
            &input_array,
            &normalized,
            &mut attribution,
        )
        .expect("executor combine should run");
    combined.evaluate().expect("combined should evaluate");
    let combined_values = combined.to_vec_f32().expect("combined copies");
    assert_f32_close(
        &combined_values,
        &[
            1.550_039_821,
            1.449_960_179,
            4.100_079_643,
            6.200_159_285,
            5.722_210_790,
            5.277_789_210,
            8.444_421_579,
            10.888_843_158,
        ],
        5.0e-5,
        "executor combine must match the pinned injection values",
    );
}

#[tokio::test]
async fn should_handle_multi_token_batches_with_per_token_rows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let executor = HyperConnectionExecutor::new(executor_plan());
    let mut values = DeterministicValues::new(0x6DAF);
    let token_count: usize = 5;
    let hyper_input: Vec<f32> = (0..token_count)
        .flat_map(|token| {
            let mut row = values.vec(8, 1.0);
            row[0] += token as f32;
            row
        })
        .collect();
    let norm_weights = values.vec(8, 0.25);
    let down = values.vec(3 * 8, 0.5);
    let up = values.vec(8 * 3, 0.5);
    let input_array =
        f32_array(&runtime, &hyper_input, &[token_count as i32, 8]).expect("input constructs");
    let norm_array = f32_array(&runtime, &norm_weights, &[8]).expect("norm constructs");
    let down_array = f32_array(&runtime, &down, &[3, 8]).expect("down constructs");
    let up_array = f32_array(&runtime, &up, &[8, 3]).expect("up constructs");
    let mut attribution = astronomical_model_serving::PerformanceAttribution::disabled();
    let (mixed, _) = executor
        .gated_mix(
            &runtime,
            &norm_array,
            &down_array,
            &up_array,
            &input_array,
            &mut attribution,
        )
        .expect("batched mixing should run");
    mixed.evaluate().expect("mixed should evaluate");
    let shape = mixed.shape();
    assert_eq!(
        shape,
        vec![token_count as i32, 4],
        "batched mixing must return one block input per token"
    );
    // Row independence: each token's mixed input must match a single-token
    // run of the same row, proving no cross-token leakage through the gates.
    let mixed_values = mixed.to_vec_f32().expect("mixed copies");
    for token in 0..token_count {
        let row_start = token * 8;
        let row: Vec<f32> = hyper_input[row_start..row_start + 8].to_vec();
        let row_array = f32_array(&runtime, &row, &[1, 8]).expect("row constructs");
        let (row_mixed, _) = executor
            .gated_mix(
                &runtime,
                &norm_array,
                &down_array,
                &up_array,
                &row_array,
                &mut attribution,
            )
            .expect("row mixing should run");
        row_mixed.evaluate().expect("row mixed should evaluate");
        let row_values = row_mixed.to_vec_f32().expect("row copies");
        assert_f32_close(
            &mixed_values[token * 4..(token + 1) * 4],
            &row_values
                .iter()
                .map(|value| *value as f64)
                .collect::<Vec<_>>(),
            1.0e-4,
            &format!("token {token} must mix independently"),
        );
    }
}

#[tokio::test]
async fn should_reject_shapes_that_disagree_with_the_plan() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let executor = HyperConnectionExecutor::new(executor_plan());
    let (hyper_input, norm_weights, down, up, _) = contract_inputs();
    let short_input =
        f32_array(&runtime, &hyper_input[..7], &[1, 7]).expect("short input constructs");
    let norm_array = f32_array(&runtime, &norm_weights, &[8]).expect("norm constructs");
    let down_array = f32_array(&runtime, &down, &[3, 8]).expect("down constructs");
    let up_array = f32_array(&runtime, &up, &[8, 3]).expect("up constructs");
    let mut attribution = astronomical_model_serving::PerformanceAttribution::disabled();
    let error = executor
        .gated_mix(
            &runtime,
            &norm_array,
            &down_array,
            &up_array,
            &short_input,
            &mut attribution,
        )
        .expect_err("a short hyper input must fail");
    assert!(
        error.to_string().contains("must hold 8 elements"),
        "the error should name the expected width: {error}"
    );
    // A block output that does not end in the stream width must fail.
    let input_array = f32_array(&runtime, &hyper_input, &[1, 8]).expect("input constructs");
    let inject = vec![1.0_f32; 16];
    let inject_array = f32_array(&runtime, &inject, &[2, 8]).expect("inject constructs");
    let wide_block = f32_array(&runtime, &vec![0.0_f32; 5], &[1, 5]).expect("block constructs");
    let normalized = executor
        .grouped_normalize(&runtime, &norm_array, &input_array, &mut attribution)
        .expect("normalization should run");
    let error = executor
        .gated_combine(
            &runtime,
            &inject_array,
            &wide_block,
            &input_array,
            &normalized,
            &mut attribution,
        )
        .expect_err("a mismatched block output must fail");
    assert!(
        error.to_string().contains("must end in 4 elements"),
        "the error should name the block width: {error}"
    );
}

#[tokio::test]
async fn should_execute_the_average_variant_with_matching_shapes() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = oracle_test_runtime();
    let executor = HyperConnectionExecutor::new(executor_plan());
    let (hyper_input, _, _, _, _) = contract_inputs();
    let input_array = f32_array(&runtime, &hyper_input, &[1, 8]).expect("input constructs");
    let mut attribution = astronomical_model_serving::PerformanceAttribution::disabled();
    let mixed = executor
        .average_mix(&runtime, &input_array, &mut attribution)
        .expect("average mixing should run");
    mixed.evaluate().expect("mixed should evaluate");
    let mixed_values = mixed.to_vec_f32().expect("mixed copies");
    assert_f32_close(
        &mixed_values,
        &[3.0, 4.0, 5.0, 6.0],
        1.0e-6,
        "average mixing must pool the streams",
    );
    let block_array =
        f32_array(&runtime, &vec![0.5, -0.5, 1.0, 2.0], &[1, 4]).expect("block constructs");
    let combined = executor
        .average_combine(&runtime, &block_array, &input_array, &mut attribution)
        .expect("average combine should run");
    combined.evaluate().expect("combined should evaluate");
    let combined_values = combined.to_vec_f32().expect("combined copies");
    assert_f32_close(
        &combined_values,
        &[1.5, 1.5, 4.0, 6.0, 5.5, 5.5, 8.0, 10.0],
        1.0e-6,
        "average combine must broadcast the block output",
    );
}
