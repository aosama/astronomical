use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
    PersistentPromptCacheBlockKey, PersistentPromptCacheModelContract,
};

#[test]
fn should_chain_blocks_under_the_resolved_storage_contract() {
    let persistent_prompt_cache_model_contract = synthetic_model_contract("model", "revision", 8);
    let parent_block_tokens = vec![10_u32; 8];
    let parent_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract,
        &parent_block_tokens,
    )
    .expect("the parent block should hash");
    let child_persistent_prompt_cache_block_key = parent_persistent_prompt_cache_block_key
        .for_child_block(&[20_u32; 8])
        .expect("the child block should hash");

    assert_eq!(parent_persistent_prompt_cache_block_key.block_index(), 0);
    assert_eq!(child_persistent_prompt_cache_block_key.block_index(), 1);
    assert_eq!(
        child_persistent_prompt_cache_block_key.block_token_count(),
        8
    );
    assert_ne!(
        parent_persistent_prompt_cache_block_key.block_hash(),
        child_persistent_prompt_cache_block_key.block_hash()
    );
}

#[test]
fn should_reject_tokens_above_the_contract_derived_block_size() {
    let persistent_prompt_cache_model_contract = synthetic_model_contract("model", "revision", 4);

    let oversized_block = PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract,
        &[1_u32; 5],
    );

    assert!(oversized_block.is_err());
}

#[test]
fn should_isolate_identical_tokens_across_model_layouts_and_policies() {
    let first_model_contract = synthetic_model_contract("model", "revision", 4);
    let second_model_contract = synthetic_model_contract("model", "revision", 8);
    let block_tokens = [1_u32, 2, 3, 4];

    let first_block_key =
        PersistentPromptCacheBlockKey::for_root_block(&first_model_contract, &block_tokens)
            .expect("the first block should hash");
    let second_block_key =
        PersistentPromptCacheBlockKey::for_root_block(&second_model_contract, &block_tokens)
            .expect("the second block should hash");

    assert_ne!(first_block_key.block_hash(), second_block_key.block_hash());
}

#[test]
fn should_bind_ordered_image_digests_only_at_the_root() {
    let persistent_prompt_cache_model_contract = synthetic_model_contract("model", "revision", 8);
    let first_image_digest = [1_u8; 32];
    let second_image_digest = [2_u8; 32];
    let block_tokens = [1_u32, 2, 3, 4];

    let first_order_block_key = PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
        &persistent_prompt_cache_model_contract,
        &block_tokens,
        &[first_image_digest, second_image_digest],
    )
    .expect("the first image order should hash");
    let second_order_block_key = PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
        &persistent_prompt_cache_model_contract,
        &block_tokens,
        &[second_image_digest, first_image_digest],
    )
    .expect("the second image order should hash");

    assert_ne!(
        first_order_block_key.block_hash(),
        second_order_block_key.block_hash()
    );
}

fn synthetic_model_contract(
    model_id: &str,
    model_revision: &str,
    attention_capacity_growth_tokens: usize,
) -> PersistentPromptCacheModelContract {
    let decoder_cache_layout =
        DecoderCacheLayout::new(vec![DecoderCacheLayerLayout::append_only_attention(
            DecoderCacheTensorLayout::sequence(
                "attention.keys",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            DecoderCacheTensorLayout::sequence(
                "attention.values",
                DecoderCacheTensorDtype::Float16,
                vec![1, 0, 2],
                1,
            ),
            attention_capacity_growth_tokens,
        )])
        .expect("the synthetic sequence layout should be valid");
    PersistentPromptCacheModelContract::resolve(
        model_id.to_owned(),
        model_revision.to_owned(),
        decoder_cache_layout,
        128,
        1_000_000,
        1_000_000,
    )
    .expect("the synthetic storage contract should resolve")
}
