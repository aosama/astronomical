use std::collections::HashMap;

use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
    PersistentPromptCacheBlockKey, PersistentPromptCacheModelContract,
};

use super::persistent_prompt_cache_disk_store_support::{
    open_persistent_prompt_cache_disk_store_with_contract, runtime_with_shared_limits,
    synthetic_tensors_for_contract,
};

#[tokio::test]
async fn should_reopen_every_sequence_only_block_in_the_complete_active_chain() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let model_contract = sequence_only_contract();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store_with_contract(
        &cache_directory,
        1_000_000,
        model_contract.clone(),
    )
    .expect("the sequence-only cache should open");
    let sequence_state_tensors = synthetic_tensors_for_contract(
        &runtime,
        &model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts(),
        model_contract.block_token_count(),
    );
    let complete_block_count = model_contract
        .maximum_context_token_count()
        .div_ceil(model_contract.block_token_count());
    publish_complete_chain(
        &cache,
        &runtime,
        &model_contract,
        complete_block_count,
        &sequence_state_tensors,
        &HashMap::new(),
    );
    drop(cache);

    let reopened_cache = open_persistent_prompt_cache_disk_store_with_contract(
        &cache_directory,
        1_000_000,
        model_contract,
    )
    .expect("the sequence-only chain should reopen");
    assert_eq!(
        reopened_cache.sequence_state_block_count(),
        complete_block_count
    );
    assert_eq!(reopened_cache.boundary_state_snapshot_count(), 0);
}

#[tokio::test]
async fn should_reopen_every_boundary_only_block_without_invalid_compaction() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = runtime_with_shared_limits();
    let model_contract = boundary_only_contract();
    let cache_directory = tempfile::tempdir().expect("the test should create a cache directory");
    let cache = open_persistent_prompt_cache_disk_store_with_contract(
        &cache_directory,
        10_000,
        model_contract.clone(),
    )
    .expect("the boundary-only cache should open");
    let boundary_state_tensors = synthetic_tensors_for_contract(
        &runtime,
        &model_contract
            .decoder_cache_layout()
            .boundary_tensor_layouts(),
        model_contract.block_token_count(),
    );
    let complete_block_count = model_contract
        .maximum_context_token_count()
        .div_ceil(model_contract.block_token_count());
    publish_complete_chain(
        &cache,
        &runtime,
        &model_contract,
        complete_block_count,
        &HashMap::new(),
        &boundary_state_tensors,
    );
    drop(cache);

    let reopened_cache = open_persistent_prompt_cache_disk_store_with_contract(
        &cache_directory,
        10_000,
        model_contract,
    )
    .expect("the boundary-only chain should reopen");
    assert_eq!(reopened_cache.sequence_state_block_count(), 0);
    assert_eq!(
        reopened_cache.boundary_state_snapshot_count(),
        complete_block_count
    );
}

fn publish_complete_chain(
    cache: &astronomical_model_serving::PersistentPromptCacheDiskStore,
    runtime: &astronomical_runtime_integration::MlxRuntime,
    model_contract: &PersistentPromptCacheModelContract,
    complete_block_count: usize,
    sequence_state_tensors: &HashMap<String, astronomical_runtime_integration::MlxArray>,
    boundary_state_tensors: &HashMap<String, astronomical_runtime_integration::MlxArray>,
) {
    let mut parent_block_key: Option<PersistentPromptCacheBlockKey> = None;
    for block_index in 0..complete_block_count {
        let block_tokens = (0..model_contract.block_token_count())
            .map(|token_offset| {
                u32::try_from(block_index * model_contract.block_token_count() + token_offset)
                    .expect("the synthetic token should fit u32")
            })
            .collect::<Vec<_>>();
        let block_key = match parent_block_key.as_ref() {
            Some(parent_block_key) => parent_block_key
                .for_child_block(&block_tokens)
                .expect("the child block identity should resolve"),
            None => PersistentPromptCacheBlockKey::for_root_block(model_contract, &block_tokens)
                .expect("the root block identity should resolve"),
        };
        cache
            .publish_block(
                runtime,
                &block_key,
                parent_block_key.as_ref(),
                sequence_state_tensors,
                boundary_state_tensors,
            )
            .expect("every complete active-chain block should publish");
        parent_block_key = Some(block_key);
    }
}

fn sequence_only_contract() -> PersistentPromptCacheModelContract {
    let decoder_cache_layout =
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 4],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 4],
                1,
            ),
            16,
        )])
        .expect("the sequence-only layout should be valid");
    PersistentPromptCacheModelContract::resolve(
        "fictional-sequence-only-model".to_owned(),
        "fictional-revision".to_owned(),
        decoder_cache_layout,
        128,
        1_000_000,
        1_000_000,
    )
    .expect("the sequence-only contract should resolve")
}

fn boundary_only_contract() -> PersistentPromptCacheModelContract {
    let decoder_cache_layout =
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::recurrent_tensor(
            DecoderCacheTensorLayout::fixed(
                "recurrent.state",
                DecoderCacheTensorDtype::Float32,
                vec![25],
            ),
        )])
        .expect("the boundary-only layout should be valid");
    PersistentPromptCacheModelContract::resolve(
        "fictional-boundary-only-model".to_owned(),
        "fictional-revision".to_owned(),
        decoder_cache_layout,
        100,
        1_000_000,
        10_000,
    )
    .expect("the boundary-only contract should resolve")
}
