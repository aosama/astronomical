use astronomical_model_serving::{
    ConvolutionState, DecoderCacheLayerLayout, DecoderCacheState, DecoderCacheTensorDtype,
    DecoderCacheTensorLayout, FullAttentionKeyValueState, GatedDeltaRecurrentState,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint, RequestDecoderStateStack,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use std::collections::HashMap;
use tokio::sync::MutexGuard;

use crate::common::qwen3_5_moe::{certified_ornith_config, persistent_prompt_cache_model_contract};
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
        .expect("the in-memory state test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    (direct_mlx_guard, runtime)
}

#[tokio::test]
async fn should_restore_attention_only_state_at_a_three_row_verifier_boundary() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let decoder_cache_layout = astronomical_model_serving::DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::append_only_attention(
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
        ),
    ])
    .expect("the attention-only cache layout should validate");
    let mut request_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("the attention-only request state should initialize");
    let DecoderCacheState::AppendOnlyAttention { attention } = request_state
        .layer_mut(0)
        .expect("the request should retain its attention layer")
    else {
        panic!("the synthetic layer should use append-only attention");
    };
    let keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the verifier keys should be valid");
    let values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the verifier values should be valid");
    attention
        .update_and_fetch(&runtime, &keys, &values, 0)
        .expect("the verifier update should populate attention state");

    request_state
        .restore_verified_prefix(
            3,
            Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                completed_prefill_chunck_tokens: 3,
                recurrent_snapshot_tensors: HashMap::new(),
            },
        )
        .expect("a three-row verifier boundary should restore exactly");

    let DecoderCacheState::AppendOnlyAttention { attention } = request_state
        .layer(0)
        .expect("the request should retain its attention layer")
    else {
        panic!("the synthetic layer should use append-only attention");
    };
    assert_eq!(attention.offset_tokens(), 3);
}

#[tokio::test]
async fn should_grow_full_attention_kv_capacity_in_256_token_steps() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut kv_state = FullAttentionKeyValueState::empty_with_growth_tokens(256)
        .expect("the test attention growth should be valid");
    assert!(
        kv_state.capacity_tokens() == 0,
        "an empty KV state should report zero capacity before the first update"
    );

    // First update of 256 tokens allocates the first 256-token slab exactly.
    let new_keys = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the first new keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the first new values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 0)
        .expect("the first KV update should grow capacity and return active views");
    assert_eq!(
        kv_state.capacity_tokens(),
        256,
        "a 256-token first update should allocate exactly a 256-token slab"
    );
    assert_eq!(kv_state.offset_tokens(), 256);

    // Second update of 256 tokens grows the slab by exactly one more 256-token step.
    let new_keys = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the second new keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the second new values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 256)
        .expect("the second KV update should grow the slab by one step");
    assert_eq!(
        kv_state.capacity_tokens(),
        512,
        "a 256-token second update at a step boundary should grow capacity to 512"
    );
    assert_eq!(kv_state.offset_tokens(), 512);

    // Third update of 256 tokens grows the slab by exactly one more 256-token step.
    let new_keys = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the third new keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 131_072], &[1, 2, 256, 256])
        .expect("the third new values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 512)
        .expect("the third KV update should grow the slab by one more step");
    assert_eq!(
        kv_state.capacity_tokens(),
        768,
        "a 256-token third update at a step boundary should grow capacity to 768"
    );
    assert_eq!(kv_state.offset_tokens(), 768);
}

#[test]
fn should_create_qwen_request_state_from_the_shared_decoder_cache_layout() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();

    let request_decoder_state = RequestDecoderStateStack::empty_from_decoder_cache_layout(
        persistent_prompt_cache_model_contract.decoder_cache_layout(),
    )
    .expect("the certified Qwen decoder-cache layout should create request state");

    assert_eq!(
        request_decoder_state.layer_count(),
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .layer_count()
    );
    assert!(matches!(
        request_decoder_state.layer(3),
        Some(DecoderCacheState::AppendOnlyAttention { .. })
    ));
    assert!(matches!(
        request_decoder_state.layer(0),
        Some(DecoderCacheState::Composite { .. })
    ));
}

#[tokio::test]
async fn should_grow_full_attention_kv_capacity_with_configured_token_steps() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut kv_state = FullAttentionKeyValueState::empty_with_growth_tokens(4)
        .expect("a positive configured KV-state growth token count should be accepted");

    let new_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the first tiny keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the first tiny values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 0)
        .expect("the first configured KV update should allocate one configured step");
    assert_eq!(kv_state.capacity_tokens(), 4);
    assert_eq!(kv_state.offset_tokens(), 4);

    let new_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the second tiny keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the second tiny values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 4)
        .expect("the second configured KV update should allocate another configured step");

    assert_eq!(
        kv_state.capacity_tokens(),
        8,
        "a configured 4-token growth step should grow capacity to 8 after two exact-step updates"
    );
    assert_eq!(kv_state.offset_tokens(), 8);
}

#[tokio::test]
async fn should_restore_full_attention_allocation_checkpoint_after_failed_growth() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut kv_state = FullAttentionKeyValueState::empty_with_growth_tokens(4)
        .expect("a positive KV-state growth step should create empty state");
    let initial_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the initial keys tensor should be valid");
    let initial_values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the initial values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial KV update should allocate one slab");
    let allocation_checkpoint = kv_state
        .allocation_checkpoint()
        .expect("the allocated KV state should be checkpointable");

    let retry_keys = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the retry keys tensor should be valid");
    let retry_values = runtime
        .array_from_f32(&[0.0; 4], &[1, 1, 4, 1])
        .expect("the retry values tensor should be valid");
    kv_state
        .update_and_fetch(&runtime, &retry_keys, &retry_values, 4)
        .expect("the failed-attempt stand-in should grow the KV state");
    assert_eq!(kv_state.capacity_tokens(), 8);

    kv_state
        .restore_allocation_checkpoint(allocation_checkpoint)
        .expect("the KV state should restore its prior physical owners");
    assert_eq!(kv_state.capacity_tokens(), 4);
    assert_eq!(kv_state.offset_tokens(), 4);
}

#[test]
fn should_project_first_use_composite_state_from_validated_tensor_layout() {
    let decoder_cache_layout = astronomical_model_serving::DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::composite(vec![
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
        ]),
    ])
    .expect("the synthetic composite decoder-cache layout should validate");
    let request_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("the synthetic composite decoder state should be constructible");

    let first_use_growth_bytes = request_decoder_state
        .projected_persistent_state_growth_bytes(&decoder_cache_layout, 1)
        .expect("first-use composite state growth should be projectable");

    assert_eq!(first_use_growth_bytes, 60);
}

#[test]
fn should_project_zero_fixed_state_growth_after_composite_state_is_materialized() {
    let decoder_cache_layout = astronomical_model_serving::DecoderCacheLayout::new(vec![
        DecoderCacheLayerLayout::composite(vec![
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
        ]),
    ])
    .expect("the synthetic composite decoder-cache layout should validate");
    let request_decoder_state =
        RequestDecoderStateStack::empty_from_decoder_cache_layout(&decoder_cache_layout)
            .expect("the synthetic composite decoder state should be constructible");

    let warm_state_growth_bytes = request_decoder_state
        .projected_persistent_state_growth_bytes(&decoder_cache_layout, 0)
        .expect("warm composite state growth should be projectable");

    assert_eq!(warm_state_growth_bytes, 60);
}

#[tokio::test]
async fn should_return_active_view_covering_only_written_tokens() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut kv_state = FullAttentionKeyValueState::empty_with_growth_tokens(256)
        .expect("the test attention growth should be valid");

    let new_keys = runtime
        .array_from_f32(&[0.0; 51_200], &[1, 2, 100, 256])
        .expect("the new keys tensor should be valid");
    let new_values = runtime
        .array_from_f32(&[0.0; 51_200], &[1, 2, 100, 256])
        .expect("the new values tensor should be valid");
    let (active_keys, active_values) = kv_state
        .update_and_fetch(&runtime, &new_keys, &new_values, 0)
        .expect("the first KV update should return active views");

    assert_eq!(
        active_keys.shape(),
        vec![1, 2, 100, 256],
        "the active keys view should cover exactly the 100 written tokens, not the 256-token capacity"
    );
    assert_eq!(
        active_values.shape(),
        vec![1, 2, 100, 256],
        "the active values view should cover exactly the 100 written tokens"
    );
}

#[tokio::test]
async fn should_restore_decoder_state_stack_to_checkpoint_after_mtp_attention_update() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let ornith_config = certified_ornith_config();
    let full_attention_layer_index = (0..ornith_config.layer_count() as usize)
        .find(|layer_index| ornith_config.decoder_layer_is_full_attention(*layer_index))
        .expect("the certified config should contain at least one full-attention layer");
    let mut decoder_state_stack = crate::common::standard_request_decoder_state(&ornith_config);

    let initial_keys = runtime
        .array_from_f32(&[1.0], &[1, 1, 1, 1])
        .expect("the initial MTP-test keys tensor should be valid");
    let initial_values = runtime
        .array_from_f32(&[2.0], &[1, 1, 1, 1])
        .expect("the initial MTP-test values tensor should be valid");
    let DecoderCacheState::AppendOnlyAttention { attention } = decoder_state_stack
        .layer_mut(full_attention_layer_index)
        .expect("the selected full-attention layer should exist")
    else {
        panic!("the selected layer should be full-attention");
    };
    attention
        .update_and_fetch(&runtime, &initial_keys, &initial_values, 0)
        .expect("the initial full-attention update should populate the layer state");

    let decoder_state_checkpoint = decoder_state_stack
        .checkpoint()
        .expect("checkpointing a populated decoder stack should capture its logical state");

    let mtp_keys = runtime
        .array_from_f32(&[3.0], &[1, 1, 1, 1])
        .expect("the MTP keys tensor should be valid");
    let mtp_values = runtime
        .array_from_f32(&[4.0], &[1, 1, 1, 1])
        .expect("the MTP values tensor should be valid");
    let DecoderCacheState::AppendOnlyAttention { attention } = decoder_state_stack
        .layer_mut(full_attention_layer_index)
        .expect("the selected full-attention layer should still exist")
    else {
        panic!("the selected layer should still be full-attention");
    };
    attention
        .update_and_fetch(&runtime, &mtp_keys, &mtp_values, 1)
        .expect("the MTP full-attention update should advance the logical offset");
    assert_eq!(
        attention.offset_tokens(),
        2,
        "the MTP update should advance the full-attention offset"
    );

    decoder_state_stack
        .restore_checkpoint(decoder_state_checkpoint)
        .expect("restoring the decoder checkpoint should discard MTP state");
    let DecoderCacheState::AppendOnlyAttention { attention } = decoder_state_stack
        .layer(full_attention_layer_index)
        .expect("the selected full-attention layer should remain available after restore")
    else {
        panic!("the selected layer should remain full-attention after restore");
    };
    assert_eq!(
        attention.offset_tokens(),
        1,
        "restore should move the full-attention offset back to the checkpoint"
    );
    assert!(
        (0..decoder_state_stack.layer_count()).all(|layer_index| decoder_state_stack
            .layer(layer_index)
            .expect("each decoder layer should remain present")
            .tensors_are_allocated_consistently()),
        "checkpoint restore must keep every decoder layer internally consistent"
    );
}

#[tokio::test]
async fn should_lazily_allocate_gated_delta_recurrent_state_as_zeros_on_first_use() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut recurrent_state = GatedDeltaRecurrentState::empty();
    assert!(
        recurrent_state.is_unallocated(),
        "an empty recurrent state should not allocate MLX arrays before the first use"
    );

    // First use materializes a float32 zero tensor of the certified gated-delta shape.
    let recurrent_state_view = recurrent_state
        .current_or_zero(&runtime)
        .expect("the first current_or_zero call should allocate a zero tensor");
    assert!(!recurrent_state.is_unallocated());
    assert_eq!(
        recurrent_state_view.shape(),
        vec![1, 32, 128, 128],
        "the lazily allocated recurrent state should use the requested shape"
    );
    assert_eq!(
        recurrent_state_view.dtype(),
        astronomical_runtime_integration::MlxDtype::Float32,
        "the recurrent state must always be float32"
    );
}

#[tokio::test]
async fn should_roll_convolution_state_buffer_by_token_count() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut convolution_state = ConvolutionState::empty();
    assert!(
        convolution_state.is_unallocated(),
        "an empty convolution state should not allocate MLX arrays before the first use"
    );

    // Feed a 100-token mixed-query-key-value input; the rolling buffer keeps the last 3 tokens.
    let mixed_queries_keys_values = runtime
        .array_from_f32(&[1.0; 819_200], &[1, 100, 8_192])
        .expect("the mixed query/key/value input should be valid");
    convolution_state
        .update(&runtime, &mixed_queries_keys_values, 100)
        .expect("the convolution update should roll the 3-token rolling buffer");
    assert!(!convolution_state.is_unallocated());

    let next_state = convolution_state
        .state()
        .expect("the convolution state should expose its rolled buffer after the first update");
    assert_eq!(
        next_state.shape(),
        vec![1, 3, 8_192],
        "the convolution rolling buffer should keep exactly the last 3 tokens"
    );
}

#[tokio::test]
async fn should_create_empty_decoder_state_stack_in_certified_layer_order() {
    let ornith_config = certified_ornith_config();
    let decoder_layer_count = ornith_config.layer_count() as usize;
    let decoder_state_stack = crate::common::standard_request_decoder_state(&ornith_config);

    assert_eq!(
        decoder_state_stack.layer_count(),
        decoder_layer_count,
        "the decoder stack should match the certified Ornith layer count"
    );
    for layer_index in 0..decoder_layer_count {
        let layer_state = decoder_state_stack
            .layer(layer_index)
            .expect("every certified decoder layer should own one model-state entry");
        if ornith_config.decoder_layer_is_full_attention(layer_index) {
            assert!(
                matches!(layer_state, DecoderCacheState::AppendOnlyAttention { .. }),
                "layer {} should be full-attention",
                layer_index
            );
        } else {
            assert!(
                matches!(layer_state, DecoderCacheState::Composite { .. }),
                "layer {} should be linear-attention",
                layer_index
            );
        }
    }
    assert!(
        decoder_state_stack.layer(decoder_layer_count).is_none(),
        "the stack should not expose an out-of-range layer"
    );
}

#[tokio::test]
async fn should_expose_decoder_state_stack_mutably_in_the_same_decoder_order() {
    let ornith_config = certified_ornith_config();
    let decoder_layer_count = ornith_config.layer_count() as usize;
    let mut decoder_state_stack = crate::common::standard_request_decoder_state(&ornith_config);

    for layer_index in 0..decoder_layer_count {
        let layer_state = decoder_state_stack
            .layer_mut(layer_index)
            .expect("every certified decoder layer should be mutably accessible");
        if ornith_config.decoder_layer_is_full_attention(layer_index) {
            assert!(matches!(
                layer_state,
                DecoderCacheState::AppendOnlyAttention { .. }
            ));
        } else {
            assert!(matches!(layer_state, DecoderCacheState::Composite { .. }));
        }
    }
    assert!(
        decoder_state_stack.layer_mut(decoder_layer_count).is_none(),
        "mutable access should also reject out-of-range layers"
    );
}

#[tokio::test]
async fn should_report_inconsistent_linear_layer_tensor_allocation() {
    let (_direct_mlx_guard, runtime) = test_runtime().await;
    let mut decoder_state_stack =
        crate::common::standard_request_decoder_state(&certified_ornith_config());
    let linear_layer_state = decoder_state_stack
        .layer_mut(0)
        .expect("layer 0 should exist in the certified decoder stack");

    match linear_layer_state {
        DecoderCacheState::Composite {
            convolution,
            recurrent,
        } => {
            convolution.restore_from_snapshot(
                runtime
                    .zeros(
                        &[1, 3, 8_192],
                        astronomical_runtime_integration::MlxDtype::BFloat16,
                    )
                    .expect("the test should create a convolution snapshot"),
            );
            assert!(
                recurrent.is_unallocated(),
                "the test fixture must leave recurrent state absent"
            );
        }
        DecoderCacheState::AppendOnlyAttention { .. } => {
            panic!("layer 0 should be linear-attention")
        }
    }
    assert!(
        !linear_layer_state.tensors_are_allocated_consistently(),
        "a linear layer with only convolution state restored must be reported inconsistent"
    );
}
