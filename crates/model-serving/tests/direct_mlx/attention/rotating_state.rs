use astronomical_model_serving::{
    FullAttentionKeyValueState, PerformanceAttribution, PerformanceOperation,
    RotatingKeyValueState, rotating_prefill_transient_token_count,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

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
    let update_measurement = attribution
        .operation_measurement(PerformanceOperation::RotatingKeyValueStateUpdate)
        .expect("enabled attribution should retain rotating-update boundaries");
    assert_eq!(update_measurement.occurrence_count(), 3);
    assert!(
        update_measurement.last_ended_offset_nanoseconds()
            >= update_measurement.first_started_offset_nanoseconds()
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

#[tokio::test]
async fn should_preserve_key_value_pairs_and_attention_across_ring_wrap() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut state = RotatingKeyValueState::empty(4).expect("rotating state");
    let mut attribution = PerformanceAttribution::disabled();
    let initial_window = token_tensor(&runtime, &[1.0, 2.0, 3.0, 4.0]);
    let (initial_attention, _) = state
        .update_and_fetch(&runtime, &initial_window, &initial_window, &mut attribution)
        .expect("initial window should commit");
    assert_eq!(token_values(&initial_attention), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(state.ring_write_index(), 4);

    let fifth_token = token_tensor(&runtime, &[5.0]);
    let (first_wrapped_attention, _) = state
        .update_and_fetch(&runtime, &fifth_token, &fifth_token, &mut attribution)
        .expect("first wrapped token should commit");
    assert_eq!(
        token_values(&first_wrapped_attention),
        vec![5.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(state.ring_write_index(), 1);

    let sixth_token = token_tensor(&runtime, &[6.0]);
    let (second_wrapped_attention, _) = state
        .update_and_fetch(&runtime, &sixth_token, &sixth_token, &mut attribution)
        .expect("second wrapped token should commit");
    assert_eq!(
        token_values(&second_wrapped_attention),
        vec![5.0, 6.0, 3.0, 4.0]
    );
    assert_eq!(state.ring_write_index(), 2);
    // One-token decode uses unmasked attention, which is invariant when key and
    // value rows undergo the same ring permutation. Prove the physical slot
    // update remains equivalent to the chronological logical window without
    // adding a hot-path reorder.
    let query = token_tensor(&runtime, &[1.0]);
    let chronological_window = token_tensor(&runtime, &[3.0, 4.0, 5.0, 6.0]);
    let physical_output = runtime
        .scaled_dot_product_attention(
            &query,
            &second_wrapped_attention,
            &second_wrapped_attention,
            1.0,
        )
        .expect("physical ring attention should execute");
    let chronological_output = runtime
        .scaled_dot_product_attention(&query, &chronological_window, &chronological_window, 1.0)
        .expect("chronological reference attention should execute");
    let physical_value = token_values(&physical_output)[0];
    let chronological_value = token_values(&chronological_output)[0];
    assert!(
        (physical_value - chronological_value).abs() <= 1e-5,
        "ring permutation changed attention from {chronological_value} to {physical_value}"
    );
    assert!(
        attribution
            .operation_measurement(PerformanceOperation::RotatingKeyValueStateUpdate)
            .is_none(),
        "disabled attribution must not retain optional measurements"
    );
}

#[tokio::test]
async fn should_keep_only_prior_window_plus_current_chunk_in_temporal_order() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let mut state = RotatingKeyValueState::empty(4).expect("rotating state");
    let mut attribution = PerformanceAttribution::enabled();
    let first_chunk = token_tensor(&runtime, &[1.0, 2.0, 3.0]);
    state
        .update_and_fetch(&runtime, &first_chunk, &first_chunk, &mut attribution)
        .expect("first chunk should commit");
    let second_chunk = token_tensor(&runtime, &[4.0, 5.0, 6.0]);
    let (second_attention, _) = state
        .update_and_fetch(&runtime, &second_chunk, &second_chunk, &mut attribution)
        .expect("second chunk should expose the bounded transient");
    assert_eq!(
        token_values(&second_attention),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(state.committed_token_count(), 4);
    let third_chunk = token_tensor(&runtime, &[7.0, 8.0]);
    let (third_attention, _) = state
        .update_and_fetch(&runtime, &third_chunk, &third_chunk, &mut attribution)
        .expect("third chunk should discard only tokens outside its transient");
    assert_eq!(
        token_values(&third_attention),
        vec![4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(state.absolute_position(), 8);
    assert_eq!(state.committed_token_count(), 4);
}

#[tokio::test]
async fn should_reject_mismatched_dtype_and_invalid_restored_counter_geometry() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let float32_keys = token_tensor(&runtime, &[1.0, 2.0]);
    let float16_values = runtime
        .astype(&float32_keys, MlxDtype::Float16)
        .expect("Float16 values should build");
    let mut state = RotatingKeyValueState::empty(4).expect("rotating state");
    state
        .update_and_fetch(
            &runtime,
            &float32_keys,
            &float16_values,
            &mut PerformanceAttribution::disabled(),
        )
        .expect_err("rotating keys and values with different dtypes must fail");

    let restored_keys = token_tensor(&runtime, &[1.0, 2.0]);
    let restored_values = token_tensor(&runtime, &[1.0, 2.0]);
    state
        .restore_from_blocks(restored_keys, restored_values, 1, 2)
        .expect_err("two restored tokens cannot claim one absolute position");

    let restored_keys = token_tensor(&runtime, &[1.0, 2.0]);
    let restored_values = token_tensor(&runtime, &[1.0, 2.0]);
    state
        .restore_from_blocks(restored_keys, restored_values, 2, 1)
        .expect_err("a growing restored state must write after its final committed token");
}

fn token_tensor(runtime: &MlxRuntime, values: &[f32]) -> MlxArray {
    runtime
        .array_from_f32(values, &[1, 1, values.len() as i32, 1])
        .expect("token tensor")
}

fn token_values(tokens: &MlxArray) -> Vec<f32> {
    tokens
        .to_vec_f32()
        .expect("rotating attention values should evaluate")
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
