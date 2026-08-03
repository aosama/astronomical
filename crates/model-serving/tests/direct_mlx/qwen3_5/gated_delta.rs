use astronomical_model_serving::{
    qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_step,
};
use astronomical_runtime_integration::MlxArray;
use astronomical_runtime_integration::{MlxCompiledElementwiseGraphs, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_apply_one_ops_based_gated_delta_recurrent_step() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let queries = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 1, 2])
        .expect("the queries should be valid");
    let keys = runtime
        .array_from_f32(&[0.5, 1.0], &[1, 1, 2])
        .expect("the keys should be valid");
    let values = runtime
        .array_from_f32(&[4.0, 8.0], &[1, 2, 1])
        .expect("the values should be valid");
    let decays = runtime
        .array_from_f32(&[0.5, 0.25], &[1, 2])
        .expect("the decays should be valid");
    let update_rates = runtime
        .array_from_f32(&[0.5, 0.25], &[1, 2])
        .expect("the update rates should be valid");
    let recurrent_state = runtime
        .array_from_f32(&[2.0, 1.0, 1.0, 3.0], &[1, 2, 1, 2])
        .expect("the recurrent state should be valid");

    let (output_values, next_recurrent_state) = qwen3_5_gated_delta_step(
        &runtime,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        &recurrent_state,
    )
    .expect("the gated-delta step should build a valid graph");

    assert_f32_close(
        &output_values
            .to_vec_f32()
            .expect("the output should evaluate as float32"),
        &[5.75, 6.203_125],
    );
    assert_f32_close(
        &next_recurrent_state
            .to_vec_f32()
            .expect("the next recurrent state should evaluate as float32"),
        &[1.75, 2.0, 1.140_625, 2.531_25],
    );
}

#[tokio::test]
async fn should_match_ops_loop_when_applying_fused_gated_delta_sequence() {
    const TOKEN_COUNT: i32 = 2;
    const KEY_HEAD_COUNT: i32 = 16;
    const VALUE_HEAD_COUNT: i32 = 32;
    const HEAD_DIMENSION: i32 = 128;

    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let query_values = patterned_values(
        (TOKEN_COUNT * KEY_HEAD_COUNT * HEAD_DIMENSION) as usize,
        0.001,
        -0.01,
    );
    let key_values = patterned_values(
        (TOKEN_COUNT * KEY_HEAD_COUNT * HEAD_DIMENSION) as usize,
        0.0015,
        0.02,
    );
    let value_values = patterned_values(
        (TOKEN_COUNT * VALUE_HEAD_COUNT * HEAD_DIMENSION) as usize,
        0.002,
        -0.03,
    );
    let decay_values = positive_patterned_values((TOKEN_COUNT * VALUE_HEAD_COUNT) as usize, 0.82);
    let update_rate_values =
        positive_patterned_values((TOKEN_COUNT * VALUE_HEAD_COUNT) as usize, 0.18);
    let recurrent_state_values = patterned_values(
        (VALUE_HEAD_COUNT * HEAD_DIMENSION * HEAD_DIMENSION) as usize,
        0.0002,
        0.005,
    );

    let queries = runtime
        .array_from_f32(
            &query_values,
            &[1, TOKEN_COUNT, KEY_HEAD_COUNT, HEAD_DIMENSION],
        )
        .expect("the sequence queries should be valid");
    let keys = runtime
        .array_from_f32(
            &key_values,
            &[1, TOKEN_COUNT, KEY_HEAD_COUNT, HEAD_DIMENSION],
        )
        .expect("the sequence keys should be valid");
    let values = runtime
        .array_from_f32(
            &value_values,
            &[1, TOKEN_COUNT, VALUE_HEAD_COUNT, HEAD_DIMENSION],
        )
        .expect("the sequence values should be valid");
    let decays = runtime
        .array_from_f32(&decay_values, &[1, TOKEN_COUNT, VALUE_HEAD_COUNT])
        .expect("the sequence decays should be valid");
    let update_rates = runtime
        .array_from_f32(&update_rate_values, &[1, TOKEN_COUNT, VALUE_HEAD_COUNT])
        .expect("the sequence update rates should be valid");
    let loop_initial_recurrent_state = runtime
        .array_from_f32(
            &recurrent_state_values,
            &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
        )
        .expect("the loop recurrent state should be valid");
    let fused_initial_recurrent_state = runtime
        .array_from_f32(
            &recurrent_state_values,
            &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
        )
        .expect("the fused recurrent state should be valid");

    let gated_delta_kernel =
        qwen3_5_gated_delta_kernel().expect("the fused gated-delta kernel should construct");
    let (fused_outputs, fused_recurrent_state) = qwen3_5_gated_delta_sequence(
        &runtime,
        &gated_delta_kernel,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        &fused_initial_recurrent_state,
    )
    .expect("the fused gated-delta kernel should build a valid graph");
    let (loop_outputs, loop_recurrent_state) = apply_ops_loop(
        &runtime,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        loop_initial_recurrent_state,
        TOKEN_COUNT,
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
    );

    assert_f32_close_with_tolerance(
        &fused_outputs
            .to_vec_f32()
            .expect("the fused outputs should evaluate"),
        &loop_outputs
            .to_vec_f32()
            .expect("the loop outputs should evaluate"),
        1e-3,
    );
    assert_f32_close_with_tolerance(
        &fused_recurrent_state
            .to_vec_f32()
            .expect("the fused recurrent state should evaluate"),
        &loop_recurrent_state
            .to_vec_f32()
            .expect("the loop recurrent state should evaluate"),
        1e-3,
    );
}

#[tokio::test]
async fn should_apply_fused_gated_delta_tail_after_full_persistent_prompt_cache_block() {
    const FULL_BLOCK_TOKEN_COUNT: i32 = 2_048;
    const TAIL_TOKEN_COUNT: i32 = 26;
    const KEY_HEAD_COUNT: i32 = 16;
    const VALUE_HEAD_COUNT: i32 = 32;
    const HEAD_DIMENSION: i32 = 128;

    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs = MlxCompiledElementwiseGraphs::new()
        .expect("the compiled elementwise graphs should initialize");
    let gated_delta_kernel =
        qwen3_5_gated_delta_kernel().expect("the fused gated-delta kernel should construct");
    let decay_rate_logarithm = runtime
        .array_from_f32(
            &positive_patterned_values(VALUE_HEAD_COUNT as usize, 0.1),
            &[VALUE_HEAD_COUNT],
        )
        .expect("the decay rate logarithm should be valid");
    let decay_interval_bias = runtime
        .array_from_f32(
            &positive_patterned_values(VALUE_HEAD_COUNT as usize, 0.05),
            &[VALUE_HEAD_COUNT],
        )
        .expect("the decay interval bias should be valid");
    let initial_recurrent_state = runtime
        .zeros(
            &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .expect("the initial recurrent state should be valid");

    let (full_block_outputs, full_block_recurrent_state) = apply_fused_gated_delta_chunk(
        &runtime,
        &compiled_elementwise_graphs,
        &gated_delta_kernel,
        &decay_rate_logarithm,
        &decay_interval_bias,
        FULL_BLOCK_TOKEN_COUNT,
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
        &initial_recurrent_state,
    );
    runtime
        .evaluate_arrays(&[&full_block_outputs, &full_block_recurrent_state])
        .expect("the full block recurrent state should evaluate before the tail chunk");

    let (tail_outputs, tail_recurrent_state) = apply_fused_gated_delta_chunk(
        &runtime,
        &compiled_elementwise_graphs,
        &gated_delta_kernel,
        &decay_rate_logarithm,
        &decay_interval_bias,
        TAIL_TOKEN_COUNT,
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
        &full_block_recurrent_state,
    );

    assert_eq!(
        tail_outputs.shape(),
        vec![1, TAIL_TOKEN_COUNT, VALUE_HEAD_COUNT, HEAD_DIMENSION]
    );
    assert_eq!(
        tail_recurrent_state.shape(),
        vec![1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION]
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_ops_loop(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    mut recurrent_state: MlxArray,
    token_count: i32,
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
) -> (MlxArray, MlxArray) {
    let mut token_outputs = Vec::new();
    for token_index in 0..token_count {
        let token_queries = slice_rank_four_token(
            runtime,
            queries,
            token_index,
            key_head_count,
            head_dimension,
        );
        let token_keys =
            slice_rank_four_token(runtime, keys, token_index, key_head_count, head_dimension);
        let token_values = slice_rank_four_token(
            runtime,
            values,
            token_index,
            value_head_count,
            head_dimension,
        );
        let token_decays = slice_rank_three_token(runtime, decays, token_index, value_head_count);
        let token_update_rates =
            slice_rank_three_token(runtime, update_rates, token_index, value_head_count);
        let (token_output, next_recurrent_state) = qwen3_5_gated_delta_step(
            runtime,
            &token_queries,
            &token_keys,
            &token_values,
            &token_decays,
            &token_update_rates,
            &recurrent_state,
        )
        .expect("the ops gated-delta step should build a valid graph");
        token_outputs.push(token_output);
        recurrent_state = next_recurrent_state;
    }
    let token_output_references = token_outputs.iter().collect::<Vec<_>>();
    let sequence_outputs = runtime
        .stack_axis(&token_output_references, 1)
        .expect("the loop outputs should stack");
    (sequence_outputs, recurrent_state)
}

fn slice_rank_four_token(
    runtime: &MlxRuntime,
    sequence_array: &MlxArray,
    token_index: i32,
    head_count: i32,
    head_dimension: i32,
) -> MlxArray {
    let sliced_token = runtime
        .slice(
            sequence_array,
            &[0, token_index, 0, 0],
            &[1, token_index + 1, head_count, head_dimension],
            &[1, 1, 1, 1],
        )
        .expect("the rank-four token should slice");
    runtime
        .squeeze_axis(&sliced_token, 1)
        .expect("the rank-four token axis should squeeze")
}

fn slice_rank_three_token(
    runtime: &MlxRuntime,
    sequence_array: &MlxArray,
    token_index: i32,
    head_count: i32,
) -> MlxArray {
    let sliced_token = runtime
        .slice(
            sequence_array,
            &[0, token_index, 0],
            &[1, token_index + 1, head_count],
            &[1, 1, 1],
        )
        .expect("the rank-three token should slice");
    runtime
        .squeeze_axis(&sliced_token, 1)
        .expect("the rank-three token axis should squeeze")
}

#[allow(clippy::too_many_arguments)]
fn apply_fused_gated_delta_chunk(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    gated_delta_kernel: &astronomical_runtime_integration::MlxMetalKernel,
    decay_rate_logarithm: &MlxArray,
    decay_interval_bias: &MlxArray,
    token_count: i32,
    key_head_count: i32,
    value_head_count: i32,
    head_dimension: i32,
    recurrent_state: &MlxArray,
) -> (MlxArray, MlxArray) {
    let queries = sequence_array(
        runtime,
        token_count,
        key_head_count,
        head_dimension,
        0.0001,
        -0.01,
    );
    let keys = sequence_array(
        runtime,
        token_count,
        key_head_count,
        head_dimension,
        0.00015,
        0.02,
    );
    let values = sequence_array(
        runtime,
        token_count,
        value_head_count,
        head_dimension,
        0.0002,
        -0.03,
    );
    let decay_interval_inputs = runtime
        .array_from_f32(
            &positive_patterned_values((token_count * value_head_count) as usize, 0.2),
            &[1, token_count, value_head_count],
        )
        .expect("the decay interval inputs should be valid");
    let decays = runtime
        .apply_compiled_gated_delta_decay(
            compiled_elementwise_graphs,
            decay_rate_logarithm,
            &decay_interval_inputs,
            decay_interval_bias,
        )
        .expect("the compiled gated-delta decay should preserve the chunk shape");
    assert_eq!(
        decays.shape(),
        vec![1, token_count, value_head_count],
        "compiled decays must keep the batch, token, and value-head axes"
    );
    let update_rates = runtime
        .array_from_f32(
            &positive_patterned_values((token_count * value_head_count) as usize, 0.18),
            &[1, token_count, value_head_count],
        )
        .expect("the update rates should be valid");

    qwen3_5_gated_delta_sequence(
        runtime,
        gated_delta_kernel,
        &queries,
        &keys,
        &values,
        &decays,
        &update_rates,
        recurrent_state,
    )
    .expect("the fused gated-delta chunk should accept the compiled decay shape")
}

fn sequence_array(
    runtime: &MlxRuntime,
    token_count: i32,
    head_count: i32,
    head_dimension: i32,
    scale: f32,
    offset: f32,
) -> MlxArray {
    runtime
        .array_from_f32(
            &patterned_values(
                (token_count * head_count * head_dimension) as usize,
                scale,
                offset,
            ),
            &[1, token_count, head_count, head_dimension],
        )
        .expect("the sequence array should be valid")
}

fn patterned_values(element_count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| ((value_index % 23) as f32 - 11.0) * scale + offset)
        .collect()
}

fn positive_patterned_values(element_count: usize, base_value: f32) -> Vec<f32> {
    (0..element_count)
        .map(|value_index| base_value + (value_index % 7) as f32 * 0.01)
        .collect()
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

fn assert_f32_close_with_tolerance(actual_values: &[f32], expected_values: &[f32], tolerance: f32) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        assert!(
            (*actual_value - *expected_value).abs() <= tolerance,
            "expected {actual_value} to be within {tolerance} of {expected_value}"
        );
    }
}
