use astronomical_model_serving::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
    PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey,
    PersistentPromptCacheModelContract,
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
fn should_bind_causal_inputs_at_the_block_where_they_enter_the_prompt() {
    let persistent_prompt_cache_model_contract = synthetic_model_contract("model", "revision", 8);
    let root_block_tokens = [1_u32, 2, 3, 4];
    let visual_block_tokens = [5_u32, 6, 7, 8];
    let first_visual_input = PersistentPromptCacheBlockCausalInput::from_canonical_bytes(&[1]);
    let second_visual_input = PersistentPromptCacheBlockCausalInput::from_canonical_bytes(&[2]);

    let root_block_key = PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract,
        &root_block_tokens,
    )
    .expect("the root block should hash");
    let first_visual_block_key = root_block_key
        .for_child_block_with_causal_input(&visual_block_tokens, &first_visual_input)
        .expect("the first visual child should hash");
    let second_visual_block_key = root_block_key
        .for_child_block_with_causal_input(&visual_block_tokens, &second_visual_input)
        .expect("the second visual child should hash");
    let first_descendant_block_key = first_visual_block_key
        .for_child_block(&[9_u32, 10, 11, 12])
        .expect("the first descendant should hash");
    let second_descendant_block_key = second_visual_block_key
        .for_child_block(&[9_u32, 10, 11, 12])
        .expect("the second descendant should hash");

    assert_ne!(
        first_visual_block_key.block_hash(),
        second_visual_block_key.block_hash()
    );
    assert_ne!(
        first_descendant_block_key.block_hash(),
        second_descendant_block_key.block_hash()
    );
}

#[test]
fn should_keep_text_only_identity_when_the_causal_input_is_empty() {
    let persistent_prompt_cache_model_contract = synthetic_model_contract("model", "revision", 8);
    let block_tokens = [1_u32, 2, 3, 4];

    let ordinary_block_key = PersistentPromptCacheBlockKey::for_root_block(
        &persistent_prompt_cache_model_contract,
        &block_tokens,
    )
    .expect("the ordinary block should hash");
    let explicit_empty_block_key = PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
        &persistent_prompt_cache_model_contract,
        &block_tokens,
        &PersistentPromptCacheBlockCausalInput::empty(),
    )
    .expect("the explicitly empty block should hash");

    assert_eq!(ordinary_block_key, explicit_empty_block_key);
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
        None,
        4,
    )
    .expect("the synthetic storage contract should resolve")
}
