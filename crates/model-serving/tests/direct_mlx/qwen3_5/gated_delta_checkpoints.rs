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
        Some(&checkpoint_kernel),
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
            Some(&ordinary_kernel),
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
    for (completed_prefill_chunk_tokens, checkpoint_interval_token_count) in [
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
                Some(&checkpoint_kernel),
                &complete_inputs.0,
                &complete_inputs.1,
                &complete_inputs.2,
                &complete_inputs.3,
                &complete_inputs.4,
                &initial_recurrent_state,
                &completed_prefill_chunk_tokens,
                checkpoint_interval_token_count,
            )
            .is_err(),
            "invalid checkpoint plan should be rejected: {completed_prefill_chunk_tokens:?} with interval {checkpoint_interval_token_count}"
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

#[tokio::test]
async fn should_preserve_boundary_snapshots_when_the_checkpoint_kernel_is_demoted() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let bounded_demotion_journey =
        tokio::time::timeout(std::time::Duration::from_secs(120), async {
            let runtime = test_runtime();
            let complete_inputs = gated_delta_inputs(&runtime, COMPLETE_TOKEN_COUNT);
            let initial_recurrent_state = runtime
                .zeros(
                    &[1, VALUE_HEAD_COUNT, HEAD_DIMENSION, HEAD_DIMENSION],
                    astronomical_runtime_integration::MlxDtype::Float32,
                )
                .expect("the initial recurrent state should be valid");
            let checkpoint_kernel = qwen3_5_gated_delta_checkpoint_kernel()
                .expect("the checkpoint gated-delta kernel should construct");

            let supported_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
                &runtime,
                Some(&checkpoint_kernel),
                &complete_inputs.0,
                &complete_inputs.1,
                &complete_inputs.2,
                &complete_inputs.3,
                &complete_inputs.4,
                &initial_recurrent_state,
                &[2, 4, 6],
                CHECKPOINT_INTERVAL_TOKEN_COUNT,
            )
            .expect("the supported checkpoint journey should execute");
            let demoted_result = qwen3_5_gated_delta_sequence_with_boundary_checkpoints(
                &runtime,
                None,
                &complete_inputs.0,
                &complete_inputs.1,
                &complete_inputs.2,
                &complete_inputs.3,
                &complete_inputs.4,
                &initial_recurrent_state,
                &[2, 4, 6],
                CHECKPOINT_INTERVAL_TOKEN_COUNT,
            )
            .expect("the demoted checkpoint journey should execute through the ops fallback");

            assert_close(
                &float32_values(&runtime, &demoted_result.sequence_outputs),
                &float32_values(&runtime, &supported_result.sequence_outputs),
                "demoted checkpoint sequence outputs",
            );
            assert_close(
                &float32_values(&runtime, &demoted_result.next_recurrent_state),
                &float32_values(&runtime, &supported_result.next_recurrent_state),
                "demoted checkpoint next recurrent state",
            );
            assert_eq!(
                demoted_result.recurrent_boundary_states.len(),
                supported_result.recurrent_boundary_states.len(),
                "the demoted fallback must snapshot every requested boundary position"
            );
            for (boundary_index, (demoted_state, supported_state)) in demoted_result
                .recurrent_boundary_states
                .iter()
                .zip(supported_result.recurrent_boundary_states.iter())
                .enumerate()
            {
                assert_close(
                    &float32_values(&runtime, demoted_state),
                    &float32_values(&runtime, supported_state),
                    &format!("demoted boundary state {boundary_index}"),
                );
            }
        })
        .await;
    assert!(
        bounded_demotion_journey.is_ok(),
        "the checkpoint demotion journey must finish within the 120-second bound"
    );
}

fn assert_close(actual_values: &[f32], expected_values: &[f32], description: &str) {
    assert_eq!(
        actual_values.len(),
        expected_values.len(),
        "{description} value count"
    );
    for (value_index, (actual_value, expected_value)) in
        actual_values.iter().zip(expected_values.iter()).enumerate()
    {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (actual_value - expected_value).abs() <= 1e-3 * comparison_scale,
            "{description} value {value_index} read {actual_value} but expected {expected_value}"
        );
    }
}

fn float32_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, astronomical_runtime_integration::MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
        .expect("demotion parity values should evaluate")
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the gated-delta checkpoint test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
