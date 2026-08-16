use astronomical_model_serving::Qwen3_5MtpRequestState;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::sync::MutexGuard;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

async fn test_runtime() -> (MutexGuard<'static, ()>, MlxRuntime) {
    // MLX allocator policy is process-global. Holding this shared guard keeps these ownership
    // contracts isolated from every other direct-MLX test in the integration binary.
    let direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the MTP state test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    (direct_mlx_guard, runtime)
}

#[tokio::test]
async fn should_not_grow_the_mtp_slab_when_the_next_update_fits_current_capacity() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut mtp_request_state = Qwen3_5MtpRequestState::empty_with_growth_tokens(4)
        .expect("a positive MTP growth step should create request state");
    let initial_keys = runtime
        .array_from_f32(&[0.0; 3], &[1, 1, 3, 1])
        .expect("the initial MTP keys should be valid");
    let initial_values = runtime
        .array_from_f32(&[0.0; 3], &[1, 1, 3, 1])
        .expect("the initial MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial MTP update should allocate one four-token slab");

    assert_eq!(
        mtp_request_state
            .projected_capacity_growth_tokens(1)
            .expect("an update within the MTP slab should have a valid projection"),
        0
    );

    let fitting_keys = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the fitting MTP keys should be valid");
    let fitting_values = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the fitting MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &fitting_keys, &fitting_values, 3)
        .expect("the fitting MTP update should reuse current slab capacity");

    assert_eq!(
        mtp_request_state
            .full_attention_key_value_state_mut_for_tests()
            .capacity_tokens(),
        4
    );
}

#[tokio::test]
async fn should_project_sequential_mtp_updates_across_a_slab_boundary() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut mtp_request_state = Qwen3_5MtpRequestState::empty_with_growth_tokens(4)
        .expect("a positive MTP growth step should create request state");
    let initial_keys = runtime
        .array_from_f32(&[0.0; 3], &[1, 1, 3, 1])
        .expect("the initial MTP keys should be valid");
    let initial_values = runtime
        .array_from_f32(&[0.0; 3], &[1, 1, 3, 1])
        .expect("the initial MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial MTP update should allocate one four-token slab");

    // Separate one-token predictor forwards can cross a rounded slab boundary even though the
    // same aggregate token count would appear to fit if projected as one update.
    assert_eq!(
        mtp_request_state
            .projected_sequential_capacity_growth_bytes(1, &[1, 1])
            .expect("sequential MTP updates should have a valid projection"),
        4
    );
}

#[tokio::test]
async fn should_grow_the_mtp_slab_by_one_configured_step_when_update_crosses_capacity() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut mtp_request_state = Qwen3_5MtpRequestState::empty_with_growth_tokens(4)
        .expect("a positive MTP growth step should create request state");
    let initial_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the initial MTP keys should be valid");
    let initial_values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the initial MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial MTP update should fill one four-token slab");

    assert_eq!(
        mtp_request_state
            .projected_capacity_growth_tokens(1)
            .expect("an update crossing MTP capacity should have a valid projection"),
        4
    );

    let crossing_keys = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the crossing MTP keys should be valid");
    let crossing_values = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the crossing MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &crossing_keys, &crossing_values, 4)
        .expect("the crossing MTP update should allocate one configured step");

    assert_eq!(
        mtp_request_state
            .full_attention_key_value_state_mut_for_tests()
            .capacity_tokens(),
        8
    );
}
