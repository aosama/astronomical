use astronomical_model_serving::{
    qwen3_5_gated_delta_checkpoint_kernel, qwen3_5_gated_delta_kernel,
    qwen3_5_gated_delta_sequence, qwen3_5_gated_delta_sequence_with_boundary_checkpoints,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const CHECKPOINT_INTERVAL_TOKEN_COUNT: i32 = 2;
const COMPLETE_TOKEN_COUNT: i32 = 8;
const KEY_HEAD_COUNT: i32 = 16;
const VALUE_HEAD_COUNT: i32 = 32;
const HEAD_DIMENSION: i32 = 128;

#[tokio::test]
async fn should_checkpoint_gated_delta_state_at_every_requested_boundary() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta checkpoint test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let complete_inputs = gated_delta_inputs(&runtime, COMPLETE_TOKEN_COUNT);
    let initial_recurrent_state = runtime
        .zeros(
            &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .expect("the initial recurrent state should be valid");
    let checkpoint_kernel = qwen3_5_gated_delta_checkpoint_kernel()
        .expect("the checkpoint gated-delta kernel should construct");
    let ordinary_kernel =
        qwen3_5_gated_delta_kernel().expect("the ordinary gated-delta kernel should construct");

    let checkpoint_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
        &runtime,
        &checkpoint_kernel,
        &complete_inputs.0,
        &complete_inputs.1,
        &complete_inputs.2,
        &complete_inputs.3,
        &complete_inputs.4,
        &initial_recurrent_state,
        &[2, 4, 6],
        CHECKPOINT_INTERVAL_TOKEN_COUNT,
    )
    .expect("the checkpoint gated-delta sequence should build a valid graph");

    let segment_inputs = gated_delta_inputs(&runtime, CHECKPOINT_INTERVAL_TOKEN_COUNT);
    let mut segmented_recurrent_state = initial_recurrent_state;
    let mut segmented_outputs = Vec::new();
    let mut expected_boundary_states = Vec::new();
    for segment_index in 0..4 {
        let (segment_outputs, next_segment_recurrent_state) = qwen3_5_gated_delta_sequence(
            &runtime,
            &ordinary_kernel,
            &segment_inputs.0,
            &segment_inputs.1,
            &segment_inputs.2,
            &segment_inputs.3,
            &segment_inputs.4,
            &segmented_recurrent_state,
        )
        .expect("each ordinary gated-delta segment should build a valid graph");
        segmented_outputs.push(segment_outputs);
        segmented_recurrent_state = next_segment_recurrent_state;
        if segment_index < 3 {
            expected_boundary_states.push(
                segmented_recurrent_state
                    .retain()
                    .expect("the expected boundary state should retain"),
            );
        }
    }
    let segmented_output_references = segmented_outputs.iter().collect::<Vec<_>>();
    let expected_complete_outputs = runtime
        .concatenate_axis(&segmented_output_references, 1)
        .expect("the segmented outputs should concatenate");

    assert_arrays_close(
        &checkpoint_result.sequence_outputs,
        &expected_complete_outputs,
    );
    assert_arrays_close(
        &checkpoint_result.next_recurrent_state,
        &segmented_recurrent_state,
    );
    for (checkpoint_state, expected_boundary_state) in checkpoint_result
        .recurrent_boundary_states
        .iter()
        .zip(expected_boundary_states.iter())
    {
        assert_arrays_close(checkpoint_state, expected_boundary_state);
    }
}

#[tokio::test]
async fn should_checkpoint_reject_invalid_gated_delta_boundary_plans() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta validation test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let complete_inputs = gated_delta_inputs(&runtime, COMPLETE_TOKEN_COUNT);
    let initial_recurrent_state = runtime
        .zeros(
            &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
            astronomical_runtime_integration::MlxDtype::Float32,
        )
        .expect("the validation recurrent state should be valid");
    let checkpoint_kernel = qwen3_5_gated_delta_checkpoint_kernel()
        .expect("the checkpoint gated-delta kernel should construct");
    for (completed_prefill_chunck_tokens, checkpoint_interval_token_count) in [
        (vec![], 2),
        (vec![0], 2),
        (vec![4, 2], 2),
        (vec![2, 2], 2),
        (vec![8], 2),
        (vec![9], 2),
        (vec![2, 5], 2),
        (vec![2], 0),
    ] {
        assert!(
            qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
                &runtime,
                &checkpoint_kernel,
                &complete_inputs.0,
                &complete_inputs.1,
                &complete_inputs.2,
                &complete_inputs.3,
                &complete_inputs.4,
                &initial_recurrent_state,
                &completed_prefill_chunck_tokens,
                checkpoint_interval_token_count,
            )
            .is_err(),
            "invalid checkpoint plan should be rejected: {completed_prefill_chunck_tokens:?} with interval {checkpoint_interval_token_count}"
        );
    }
}

fn gated_delta_inputs(
    runtime: &MlxRuntime,
    token_count: i32,
) -> (MlxArray, MlxArray, MlxArray, MlxArray, MlxArray) {
    let key_element_count = (token_count * KEY_HEAD_COUNT * HEAD_DIMENSION) as usize;
    let value_element_count = (token_count * VALUE_HEAD_COUNT * HEAD_DIMENSION) as usize;
    let scalar_element_count = (token_count * VALUE_HEAD_COUNT) as usize;
    (
        runtime
            .array_from_f32(
                &vec![0.01; key_element_count],
                &[1, token_count, KEY_HEAD_COUNT, HEAD_DIMENSION],
            )
            .expect("the queries should be valid"),
        runtime
            .array_from_f32(
                &vec![0.02; key_element_count],
                &[1, token_count, KEY_HEAD_COUNT, HEAD_DIMENSION],
            )
            .expect("the keys should be valid"),
        runtime
            .array_from_f32(
                &vec![0.03; value_element_count],
                &[1, token_count, VALUE_HEAD_COUNT, HEAD_DIMENSION],
            )
            .expect("the values should be valid"),
        runtime
            .array_from_f32(
                &vec![0.9; scalar_element_count],
                &[1, token_count, VALUE_HEAD_COUNT],
            )
            .expect("the decays should be valid"),
        runtime
            .array_from_f32(
                &vec![0.1; scalar_element_count],
                &[1, token_count, VALUE_HEAD_COUNT],
            )
            .expect("the update rates should be valid"),
    )
}

fn assert_arrays_close(actual: &MlxArray, expected: &MlxArray) {
    let actual_values = actual
        .to_vec_f32()
        .expect("the actual array should evaluate");
    let expected_values = expected
        .to_vec_f32()
        .expect("the expected array should evaluate");
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_scalar, expected_scalar) in actual_values.iter().zip(expected_values) {
        assert!((actual_scalar - expected_scalar).abs() <= 1e-3);
    }
}
