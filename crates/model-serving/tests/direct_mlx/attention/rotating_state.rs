use astronomical_model_serving::{
    FullAttentionKeyValueState, PerformanceAttribution, PerformanceOperation,
    RotatingKeyValueState, rotating_prefill_transient_token_count,
};
use astronomical_runtime_integration::{MlxArray, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_keep_committed_rotating_state_bounded_while_absolute_position_grows() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut rotating_state = RotatingKeyValueState::empty(4).expect("positive window");
    let mut attribution = PerformanceAttribution::enabled();
    let first_chunk = token_tensor(&runtime, &[1.0, 2.0, 3.0]);
    rotating_state
        .update_and_fetch(&runtime, &first_chunk, &first_chunk, &mut attribution)
        .expect("first prefill should append");
    let second_chunk = token_tensor(&runtime, &[4.0, 5.0]);
    let (attention_keys, _) = rotating_state
        .update_and_fetch(&runtime, &second_chunk, &second_chunk, &mut attribution)
        .expect("second prefill should expose its transient");
    assert_eq!(
        attention_keys.shape()[2] as u32,
        rotating_prefill_transient_token_count(4, 2).expect("valid transient")
    );
    assert_eq!(rotating_state.absolute_position(), 5);
    assert_eq!(rotating_state.committed_token_count(), 4);

    let decode_token = token_tensor(&runtime, &[6.0]);
    rotating_state
        .update_and_fetch(&runtime, &decode_token, &decode_token, &mut attribution)
        .expect("one-token decode should wrap");
    assert_eq!(rotating_state.absolute_position(), 6);
    assert_eq!(rotating_state.committed_token_count(), 4);
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::RotatingKeyValueStateUpdate)
            .is_some()
    );
}

#[tokio::test]
async fn should_run_append_only_and_rotating_state_in_the_same_forward() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut full_state = FullAttentionKeyValueState::empty_with_growth_tokens(8)
        .expect("append-only state should construct");
    let mut rotating_state = RotatingKeyValueState::empty(4).expect("rotating state");
    let mut attribution = PerformanceAttribution::disabled();
    let chunk = token_tensor(&runtime, &[1.0, 2.0]);
    let (full_keys, _) = full_state
        .update_and_fetch(&runtime, &chunk, &chunk, 0)
        .expect("append-only update should succeed");
    let (rotating_keys, _) = rotating_state
        .update_and_fetch(&runtime, &chunk, &chunk, &mut attribution)
        .expect("rotating update should succeed");
    assert_eq!(full_keys.shape()[2], 2);
    assert_eq!(rotating_keys.shape()[2], 2);
}

#[tokio::test]
async fn should_restore_a_rotating_allocation_checkpoint() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut state = RotatingKeyValueState::empty(4).expect("rotating state");
    let mut attribution = PerformanceAttribution::disabled();
    let first_chunk = token_tensor(&runtime, &[1.0, 2.0]);
    state
        .update_and_fetch(&runtime, &first_chunk, &first_chunk, &mut attribution)
        .expect("first update");
    let checkpoint = state.allocation_checkpoint().expect("checkpoint");
    let second_chunk = token_tensor(&runtime, &[3.0, 4.0, 5.0]);
    state
        .update_and_fetch(&runtime, &second_chunk, &second_chunk, &mut attribution)
        .expect("second update");
    state
        .restore_allocation_checkpoint(checkpoint)
        .expect("restore");
    assert_eq!(state.absolute_position(), 2);
    assert_eq!(state.committed_token_count(), 2);
}

fn token_tensor(runtime: &MlxRuntime, values: &[f32]) -> MlxArray {
    runtime
        .array_from_f32(values, &[1, 1, values.len() as i32, 1])
        .expect("token tensor")
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits"),
    )
    .expect("direct MLX runtime")
}
