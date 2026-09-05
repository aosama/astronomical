use std::collections::HashMap;

use astronomical_model_serving::{DecoderCacheState, RequestDecoderStateStack};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::qwen3_5_moe::{
    frozen_ornith_1_0_config, persistent_prompt_cache_model_contract,
};
use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_extract_split_persistent_prompt_cache_tensors_from_populated_decoder_state() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let block_token_count = persistent_prompt_cache_model_contract().block_token_count();
    let ornith_config = frozen_ornith_1_0_config();
    let mut request_decoder_state = crate::common::standard_request_decoder_state(&ornith_config);
    populate_request_decoder_state(&runtime, &mut request_decoder_state);

    let kv_block_tensors = request_decoder_state
        .extract_persistent_prompt_cache_kv_block_tensors(
            &runtime,
            0,
            block_token_count,
            block_token_count,
        )
        .expect("populated request decoder state should extract a KV block");
    let recurrent_snapshot_tensors = request_decoder_state
        .extract_persistent_prompt_cache_recurrent_snapshot_tensors()
        .expect("populated request decoder state should extract a recurrent snapshot");

    let full_attention_layer_count = (0..ornith_config.layer_count() as usize)
        .filter(|layer_index| ornith_config.decoder_layer_is_full_attention(*layer_index))
        .count();
    let linear_attention_layer_count =
        (ornith_config.layer_count() as usize) - full_attention_layer_count;
    assert_eq!(kv_block_tensors.len(), full_attention_layer_count * 2);
    assert_eq!(
        recurrent_snapshot_tensors.len(),
        linear_attention_layer_count * 2
    );
    for layer_index in 0..ornith_config.layer_count() as usize {
        let layer_is_full_attention = ornith_config.decoder_layer_is_full_attention(layer_index);
        if layer_is_full_attention {
            assert!(kv_block_tensors.contains_key(&format!("layer_{layer_index}_attention.keys")));
            assert!(
                kv_block_tensors.contains_key(&format!("layer_{layer_index}_attention.values"))
            );
            assert!(
                !recurrent_snapshot_tensors
                    .contains_key(&format!("layer_{layer_index}_attention.keys"))
            );
        } else {
            assert!(
                recurrent_snapshot_tensors
                    .contains_key(&format!("layer_{layer_index}_linear.convolution"))
            );
            assert!(
                recurrent_snapshot_tensors
                    .contains_key(&format!("layer_{layer_index}_linear.gated_delta_recurrent"))
            );
            assert!(
                !kv_block_tensors
                    .contains_key(&format!("layer_{layer_index}_linear.gated_delta_recurrent"))
            );
        }
    }
}

#[tokio::test]
async fn should_extract_only_the_requested_full_attention_kv_slice_from_longer_model_state() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let block_token_count = persistent_prompt_cache_model_contract().block_token_count();
    let ornith_config = frozen_ornith_1_0_config();
    let mut request_decoder_state = crate::common::standard_request_decoder_state(&ornith_config);
    populate_request_decoder_state_with_full_attention_tokens(
        &runtime,
        &mut request_decoder_state,
        block_token_count * 2,
    );

    let kv_block_tensors = request_decoder_state
        .extract_persistent_prompt_cache_kv_block_tensors(
            &runtime,
            block_token_count,
            block_token_count * 2,
            block_token_count,
        )
        .expect("request decoder state should extract the requested KV block slice");

    for layer_index in 0..ornith_config.layer_count() as usize {
        if ornith_config.decoder_layer_is_full_attention(layer_index) {
            let keys = kv_block_tensors
                .get(&format!("layer_{layer_index}_attention.keys"))
                .expect("the KV block should contain full-attention keys");
            let values = kv_block_tensors
                .get(&format!("layer_{layer_index}_attention.values"))
                .expect("the KV block should contain full-attention values");
            assert_eq!(keys.shape(), vec![1, 2, block_token_count as i32, 256]);
            assert_eq!(keys.shape(), values.shape());
        }
    }
}

#[tokio::test]
async fn should_restore_request_decoder_state_from_kv_blocks_and_recurrent_snapshot() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let first_kv_block_tensors = tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 10.0);
    let second_kv_block_tensors = tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 20.0);
    let recurrent_snapshot_tensors =
        tiny_persistent_prompt_cache_recurrent_snapshot_tensors(&runtime, 30.0);
    let mut kv_block_tensors = vec![first_kv_block_tensors, second_kv_block_tensors];
    let mut recurrent_snapshot_tensors = recurrent_snapshot_tensors;

    let mut restored_request_decoder_state =
        crate::common::standard_request_decoder_state(&frozen_ornith_1_0_config());
    restored_request_decoder_state
        .restore_from_persistent_prompt_cache_blocks(
            &runtime,
            &mut kv_block_tensors,
            &mut recurrent_snapshot_tensors,
        )
        .expect("split prompt-cache tensors should restore as one request decoder state");

    let full_attention_layer = restored_request_decoder_state
        .layer(3)
        .expect("layer 3 should be present in the restored request decoder state");
    assert!(kv_block_tensors.iter().all(HashMap::is_empty));
    match full_attention_layer {
        DecoderCacheState::AppendOnlyAttention { attention } => {
            let restored_keys = attention
                .keys_state()
                .expect("full-attention keys should be restored");
            let restored_values = attention
                .values_state()
                .expect("full-attention values should be restored");
            assert_eq!(attention.offset_tokens(), 4);
            assert_eq!(restored_keys.shape(), vec![1, 1, 4, 1]);
            assert_eq!(
                restored_keys
                    .to_vec_f32()
                    .expect("keys should copy back to the test"),
                vec![10.0, 11.0, 20.0, 21.0]
            );
            assert_eq!(
                restored_values
                    .to_vec_f32()
                    .expect("values should copy back to the test"),
                vec![12.0, 13.0, 22.0, 23.0]
            );
        }
        DecoderCacheState::Composite { .. } => panic!("layer 3 should be full attention"),
    }

    let linear_layer = restored_request_decoder_state
        .layer(0)
        .expect("layer 0 should be present in the restored request decoder state");
    match linear_layer {
        DecoderCacheState::Composite {
            convolution,
            recurrent,
        } => {
            assert_eq!(
                convolution
                    .state()
                    .expect("linear convolution state should be restored")
                    .to_vec_f32()
                    .expect("convolution should copy back to the test"),
                vec![34.0]
            );
            assert_eq!(
                recurrent
                    .state()
                    .expect("linear recurrent state should be restored")
                    .to_vec_f32()
                    .expect("recurrent should copy back to the test"),
                vec![35.0]
            );
        }
        DecoderCacheState::AppendOnlyAttention { .. } => panic!("layer 0 should be linear"),
    }
}

#[tokio::test]
async fn should_restore_three_kv_blocks_in_sequence_order_at_final_length() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let mut kv_block_tensors = vec![
        tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 10.0),
        tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 20.0),
        tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 30.0),
    ];
    let mut recurrent_snapshot_tensors =
        tiny_persistent_prompt_cache_recurrent_snapshot_tensors(&runtime, 40.0);

    let mut restored_request_decoder_state =
        crate::common::standard_request_decoder_state(&frozen_ornith_1_0_config());
    restored_request_decoder_state
        .restore_from_persistent_prompt_cache_blocks(
            &runtime,
            &mut kv_block_tensors,
            &mut recurrent_snapshot_tensors,
        )
        .expect("three prompt-cache blocks should restore in sequence order");

    assert!(kv_block_tensors.iter().all(HashMap::is_empty));
    let full_attention_layer = restored_request_decoder_state
        .layer(3)
        .expect("layer 3 should be present in the restored request decoder state");
    match full_attention_layer {
        DecoderCacheState::AppendOnlyAttention { attention } => {
            let restored_keys = attention
                .keys_state()
                .expect("full-attention keys should be restored");
            assert_eq!(attention.offset_tokens(), 6);
            assert_eq!(restored_keys.shape(), vec![1, 1, 6, 1]);
            assert_eq!(
                restored_keys
                    .to_vec_f32()
                    .expect("keys should copy back to the test"),
                vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
            );
        }
        DecoderCacheState::Composite { .. } => panic!("layer 3 should be full attention"),
    }
}

#[tokio::test]
async fn should_round_trip_compact_sparse_target_decoder_state_without_dense_recomputation() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let ornith_config = frozen_ornith_1_0_config();
    let mut sparse_target_decoder_state =
        crate::common::standard_request_decoder_state(&ornith_config);
    populate_request_decoder_state_with_full_attention_tokens(
        &runtime,
        &mut sparse_target_decoder_state,
        7,
    );

    let compact_target_state_tensors = sparse_target_decoder_state
        .extract_speculative_prefill_target_state_tensors(&runtime)
        .expect("sparse target state should extract every decoder layer");
    let mut restored_sparse_target_decoder_state =
        crate::common::standard_request_decoder_state(&ornith_config);
    restored_sparse_target_decoder_state
        .restore_speculative_prefill_target_state_tensors(&compact_target_state_tensors, 7)
        .expect("compact sparse target tensors should restore without target forwarding");

    let restored_attention_layer = restored_sparse_target_decoder_state
        .layer(3)
        .expect("the frozen full-attention layer should remain present");
    match restored_attention_layer {
        DecoderCacheState::AppendOnlyAttention { attention } => {
            assert_eq!(attention.offset_tokens(), 7);
            assert_eq!(
                attention
                    .keys_state()
                    .expect("restored attention keys should remain present")
                    .shape()[2],
                7
            );
        }
        DecoderCacheState::Composite { .. } => {
            panic!("the frozen full-attention layer must not change kind")
        }
    }
}

#[tokio::test]
async fn should_materialize_restored_split_persistent_prompt_cache_state_before_first_prefill() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let mut kv_block_tensors = vec![tiny_persistent_prompt_cache_kv_block_tensors(
        &runtime, 10.0,
    )];
    let mut recurrent_snapshot_tensors =
        tiny_persistent_prompt_cache_recurrent_snapshot_tensors(&runtime, 30.0);

    let mut restored_request_decoder_state =
        crate::common::standard_request_decoder_state(&frozen_ornith_1_0_config());
    restored_request_decoder_state
        .restore_from_persistent_prompt_cache_blocks(
            &runtime,
            &mut kv_block_tensors,
            &mut recurrent_snapshot_tensors,
        )
        .expect("split prompt-cache tensors should restore as one request decoder state");

    restored_request_decoder_state
        .materialize_restored_persistent_prompt_cache_state(&runtime)
        .expect("restored persistent prompt-cache tensors should be materialized before prefill");
}

#[tokio::test]
async fn should_reject_a_kv_block_tensor_map_missing_a_required_tensor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let mut kv_block_tensors = tiny_persistent_prompt_cache_kv_block_tensors(&runtime, 10.0);
    kv_block_tensors.remove("layer_3_attention.keys");
    let recurrent_snapshot_tensors =
        tiny_persistent_prompt_cache_recurrent_snapshot_tensors(&runtime, 30.0);

    let mut restored_request_decoder_state =
        crate::common::standard_request_decoder_state(&frozen_ornith_1_0_config());
    let mut kv_block_tensor_maps = [kv_block_tensors];
    let mut recurrent_snapshot_tensors = recurrent_snapshot_tensors;
    let restore_result = restored_request_decoder_state
        .restore_from_persistent_prompt_cache_blocks(
            &runtime,
            &mut kv_block_tensor_maps,
            &mut recurrent_snapshot_tensors,
        );

    assert!(restore_result.is_err());
}

#[tokio::test]
async fn should_reject_a_recurrent_snapshot_tensor_map_missing_a_required_tensor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = shared_runtime();
    let mut kv_block_tensors = vec![tiny_persistent_prompt_cache_kv_block_tensors(
        &runtime, 10.0,
    )];
    let mut recurrent_snapshot_tensors =
        tiny_persistent_prompt_cache_recurrent_snapshot_tensors(&runtime, 30.0);
    recurrent_snapshot_tensors.remove("layer_0_linear.gated_delta_recurrent");

    let mut restored_request_decoder_state =
        crate::common::standard_request_decoder_state(&frozen_ornith_1_0_config());
    let restore_result = restored_request_decoder_state
        .restore_from_persistent_prompt_cache_blocks(
            &runtime,
            &mut kv_block_tensors,
            &mut recurrent_snapshot_tensors,
        );

    assert!(restore_result.is_err());
}

fn tiny_persistent_prompt_cache_kv_block_tensors(
    runtime: &MlxRuntime,
    tensor_value_base: f32,
) -> HashMap<String, MlxArray> {
    let ornith_config = frozen_ornith_1_0_config();
    let full_attention_layer_count = (0..ornith_config.layer_count() as usize)
        .filter(|layer_index| ornith_config.decoder_layer_is_full_attention(*layer_index))
        .count();
    let mut kv_block_tensors = HashMap::with_capacity(full_attention_layer_count * 2);
    for layer_index in 0..ornith_config.layer_count() as usize {
        if ornith_config.decoder_layer_is_full_attention(layer_index) {
            kv_block_tensors.insert(
                format!("layer_{layer_index}_attention.keys"),
                runtime
                    .array_from_f32(&[tensor_value_base, tensor_value_base + 1.0], &[1, 1, 2, 1])
                    .expect("the test should create full-attention keys"),
            );
            kv_block_tensors.insert(
                format!("layer_{layer_index}_attention.values"),
                runtime
                    .array_from_f32(
                        &[tensor_value_base + 2.0, tensor_value_base + 3.0],
                        &[1, 1, 2, 1],
                    )
                    .expect("the test should create full-attention values"),
            );
        }
    }
    kv_block_tensors
}

fn tiny_persistent_prompt_cache_recurrent_snapshot_tensors(
    runtime: &MlxRuntime,
    tensor_value_base: f32,
) -> HashMap<String, MlxArray> {
    let ornith_config = frozen_ornith_1_0_config();
    let linear_attention_layer_count = (0..ornith_config.layer_count() as usize)
        .filter(|layer_index| !ornith_config.decoder_layer_is_full_attention(*layer_index))
        .count();
    let mut recurrent_snapshot_tensors = HashMap::with_capacity(linear_attention_layer_count * 2);
    for layer_index in 0..ornith_config.layer_count() as usize {
        if !ornith_config.decoder_layer_is_full_attention(layer_index) {
            recurrent_snapshot_tensors.insert(
                format!("layer_{layer_index}_linear.convolution"),
                runtime
                    .array_from_f32(&[tensor_value_base + 4.0], &[1, 1, 1])
                    .expect("the test should create linear convolution state"),
            );
            recurrent_snapshot_tensors.insert(
                format!("layer_{layer_index}_linear.gated_delta_recurrent"),
                runtime
                    .array_from_f32(&[tensor_value_base + 5.0], &[1, 1, 1])
                    .expect("the test should create linear recurrent state"),
            );
        }
    }
    recurrent_snapshot_tensors
}

fn shared_runtime() -> MlxRuntime {
    let memory_limits = MlxMemoryLimits::new(
        DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
        DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
    )
    .expect("the test memory limits should be valid");
    MlxRuntime::initialize(memory_limits).expect("the pinned MLX runtime should initialize")
}

fn populate_request_decoder_state(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
) {
    populate_request_decoder_state_with_full_attention_tokens(
        runtime,
        request_decoder_state,
        persistent_prompt_cache_model_contract().block_token_count(),
    );
}

fn populate_request_decoder_state_with_full_attention_tokens(
    runtime: &MlxRuntime,
    request_decoder_state: &mut RequestDecoderStateStack,
    full_attention_token_count: usize,
) {
    let ornith_config = frozen_ornith_1_0_config();
    let key_value_head_count = ornith_config.key_value_head_count() as i32;
    let head_dimension = ornith_config.head_dimension() as i32;
    let linear_convolution_kernel_dimension =
        ornith_config.linear_convolution_kernel_dimension() as i32;
    let linear_convolution_dimension = (ornith_config.linear_key_head_count() as usize)
        .saturating_mul(ornith_config.linear_key_head_dimension() as usize)
        .saturating_mul(2)
        .saturating_add(
            (ornith_config.linear_value_head_count() as usize)
                .saturating_mul(ornith_config.linear_value_head_dimension() as usize),
        );
    let linear_convolution_dimension_i32 = i32::try_from(linear_convolution_dimension)
        .expect("the frozen linear convolution dimension should fit i32");
    let linear_value_head_count = ornith_config.linear_value_head_count() as i32;
    let linear_value_head_dimension = ornith_config.linear_value_head_dimension() as i32;
    let linear_key_head_dimension = ornith_config.linear_key_head_dimension() as i32;
    for layer_index in 0..ornith_config.layer_count() as usize {
        let layer_model_state = request_decoder_state
            .layer_mut(layer_index)
            .expect("the layer slot should exist");
        match layer_model_state {
            DecoderCacheState::AppendOnlyAttention { attention } => {
                let attention_keys = runtime
                    .zeros(
                        &[
                            1,
                            key_value_head_count,
                            full_attention_token_count as i32,
                            head_dimension,
                        ],
                        MlxDtype::BFloat16,
                    )
                    .expect("the test should create the keys tensor");
                let attention_values = runtime
                    .zeros(
                        &[
                            1,
                            key_value_head_count,
                            full_attention_token_count as i32,
                            head_dimension,
                        ],
                        MlxDtype::BFloat16,
                    )
                    .expect("the test should create the values tensor");
                attention
                    .restore_from_blocks(attention_keys, attention_values)
                    .expect("the test KV tensors should restore into the owner");
            }
            DecoderCacheState::Composite {
                convolution,
                recurrent,
            } => {
                convolution.restore_from_snapshot(
                    runtime
                        .zeros(
                            &[
                                1,
                                linear_convolution_kernel_dimension.saturating_sub(1),
                                linear_convolution_dimension_i32,
                            ],
                            MlxDtype::BFloat16,
                        )
                        .expect("the test should create the convolution tensor"),
                );
                recurrent.restore_from_snapshot(
                    runtime
                        .zeros(
                            &[
                                1,
                                linear_value_head_count,
                                linear_value_head_dimension,
                                linear_key_head_dimension,
                            ],
                            MlxDtype::Float32,
                        )
                        .expect("the test should create the recurrent tensor"),
                );
            }
        }
    }
}
