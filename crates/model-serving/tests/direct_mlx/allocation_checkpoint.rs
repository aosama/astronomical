use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheState, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout, Qwen3_5MoEMtpRequestState, RequestDecoderStateStack,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};
use tokio::sync::MutexGuard;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

async fn test_runtime() -> (MutexGuard<'static, ()>, MlxRuntime) {
    let direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the allocation-checkpoint memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    (direct_mlx_guard, runtime)
}

fn synthetic_composite_decoder_cache_layout() -> DecoderCacheLayout {
    DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::composite(vec![
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "linear.convolution",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 3, 2],
        )),
        DecoderCacheLayerLayout::recurrent_tensor(DecoderCacheTensorLayout::fixed(
            "linear.gated_delta_recurrent",
            DecoderCacheTensorDtype::Float32,
            vec![1, 2, 2, 3],
        )),
    ])])
    .expect("the synthetic composite decoder-cache layout should validate")
}

fn synthetic_append_only_attention_decoder_cache_layout() -> DecoderCacheLayout {
    DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
        DecoderCacheTensorLayout::sequence(
            "attention.keys",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 1, 0, 1],
            2,
        ),
        DecoderCacheTensorLayout::sequence(
            "attention.values",
            DecoderCacheTensorDtype::BFloat16,
            vec![1, 1, 0, 1],
            2,
        ),
        4,
    )])
    .expect("the synthetic append-only decoder-cache layout should validate")
}

#[tokio::test]
async fn should_restore_absent_composite_owners_after_a_failed_first_use_attempt() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let decoder_cache_layout = synthetic_composite_decoder_cache_layout();
    let mut request_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("the synthetic composite state should construct");
    let allocation_checkpoint = request_decoder_state
        .allocation_checkpoint()
        .expect("an empty request stack should be checkpointable");

    let mixed_queries_keys_values = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 1, 2])
        .and_then(|float32_input| runtime.astype(&float32_input, MlxDtype::BFloat16))
        .expect("the synthetic convolution input should be valid");
    let DecoderCacheState::Composite {
        convolution,
        recurrent,
    } = request_decoder_state
        .layer_mut(0)
        .expect("the synthetic composite layer should exist")
    else {
        panic!("the synthetic layer should be composite")
    };
    convolution
        .update(&runtime, &mixed_queries_keys_values, 1)
        .expect("the first convolution update should allocate rolling state");
    recurrent
        .current_or_zero(&runtime)
        .expect("the first recurrent lookup should allocate state");
    assert_eq!(
        request_decoder_state
            .projected_persistent_state_growth_bytes(&decoder_cache_layout, 1)
            .expect("materialized composite state should project"),
        0
    );

    request_decoder_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .expect("the allocation checkpoint should restore absent composite owners");
    assert_eq!(
        request_decoder_state
            .projected_persistent_state_growth_bytes(&decoder_cache_layout, 1)
            .expect("restored empty composite state should project"),
        60
    );
}

#[tokio::test]
async fn should_restore_mtp_physical_slab_and_offset_after_failed_growth() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut mtp_request_state = Qwen3_5MoEMtpRequestState::empty_with_growth_tokens(4)
        .expect("a positive MTP growth step should create request state");
    let initial_keys = runtime
        .array_from_f32(&[1.0; 4], &[1, 1, 4, 1])
        .expect("the initial MTP keys should be valid");
    let initial_values = runtime
        .array_from_f32(&[2.0; 4], &[1, 1, 4, 1])
        .expect("the initial MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial MTP update should fill one slab");
    let allocation_checkpoint = mtp_request_state
        .allocation_checkpoint()
        .expect("the populated MTP state should be checkpointable");

    let retry_keys = runtime
        .array_from_f32(&[3.0], &[1, 1, 1, 1])
        .expect("the retry MTP keys should be valid");
    let retry_values = runtime
        .array_from_f32(&[4.0], &[1, 1, 1, 1])
        .expect("the retry MTP values should be valid");
    mtp_request_state
        .full_attention_key_value_state_mut_for_tests()
        .update_and_fetch(&runtime, &retry_keys, &retry_values, 4)
        .expect("the failed-attempt stand-in should grow the MTP slab");
    assert_eq!(
        mtp_request_state
            .full_attention_key_value_state_mut_for_tests()
            .capacity_tokens(),
        8
    );

    mtp_request_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .expect("the MTP allocation checkpoint should restore prior owners");
    assert_eq!(
        mtp_request_state
            .full_attention_key_value_state_mut_for_tests()
            .capacity_tokens(),
        4
    );
    assert_eq!(mtp_request_state.committed_token_count(), 4);
}

#[tokio::test]
async fn should_reject_allocation_checkpoint_family_mismatch_before_mutating_any_layer() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let composite_layer_layout = synthetic_composite_decoder_cache_layout()
        .layer(0)
        .expect("the synthetic composite layout should contain one layer")
        .clone();
    let append_only_layer_layout = synthetic_append_only_attention_decoder_cache_layout()
        .layer(0)
        .expect("the synthetic append-only layout should contain one layer")
        .clone();
    let live_decoder_cache_layout = DecoderCacheLayout::new(vec![
        composite_layer_layout.clone(),
        composite_layer_layout.clone(),
    ])
    .expect("the live two-layer composite layout should validate");
    let checkpoint_decoder_cache_layout =
        DecoderCacheLayout::new(vec![composite_layer_layout, append_only_layer_layout])
            .expect("the mismatched checkpoint layout should validate independently");
    let mut live_request_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&live_decoder_cache_layout)
            .expect("the live decoder state should construct");
    let checkpoint_request_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&checkpoint_decoder_cache_layout)
            .expect("the checkpoint decoder state should construct");
    let allocation_checkpoint = checkpoint_request_decoder_state
        .allocation_checkpoint()
        .expect("the mismatched source state should be checkpointable");

    let mixed_queries_keys_values = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 1, 2])
        .and_then(|float32_input| runtime.astype(&float32_input, MlxDtype::BFloat16))
        .expect("the synthetic convolution input should be valid");
    let DecoderCacheState::Composite {
        convolution,
        recurrent,
    } = live_request_decoder_state
        .layer_mut(0)
        .expect("the first live composite layer should exist")
    else {
        panic!("the first live layer should be composite")
    };
    convolution
        .update(&runtime, &mixed_queries_keys_values, 1)
        .expect("the first live convolution update should allocate rolling state");
    recurrent
        .current_or_zero(&runtime)
        .expect("the first live recurrent lookup should allocate state");

    let restoration_error = live_request_decoder_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .expect_err("a later checkpoint family mismatch must reject the complete restoration");

    assert!(
        restoration_error
            .to_string()
            .contains("families do not match")
    );
    assert_eq!(
        live_request_decoder_state
            .projected_persistent_state_growth_bytes(&live_decoder_cache_layout, 1)
            .expect("the rejected restoration should leave the live stack consistent"),
        60,
        "the first materialized layer must remain untouched when a later family mismatches"
    );
}
