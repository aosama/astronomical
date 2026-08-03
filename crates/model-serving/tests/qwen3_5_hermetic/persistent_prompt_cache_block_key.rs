use astronomical_model_serving::{
    ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID, ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
    PersistentPromptCacheBlockKey,
};

const BLOCK_TOKEN_COUNT: usize = 2_048;

#[test]
fn should_root_the_first_block_hash_at_a_fixed_model_revision_seed() {
    let first_block_tokens = vec![1_u32, 2, 3, 4];
    let persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &first_block_tokens,
    )
    .expect("the root persistent prompt-cache block identity should hash a non-empty first block");

    assert_eq!(
        persistent_prompt_cache_block_key.token_count(),
        first_block_tokens.len()
    );
    assert_eq!(persistent_prompt_cache_block_key.block_index(), 0);
    assert_ne!(persistent_prompt_cache_block_key.block_hash(), [0_u8; 32]);
    assert_ne!(
        persistent_prompt_cache_block_key.block_hash(),
        root_sentinel_hash()
    );
}

#[test]
fn should_chain_each_block_hash_off_its_parent_hash() {
    let block_tokens = vec![10_u32, 20, 30, 40];
    let parent_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the parent persistent prompt-cache block identity should hash its block");

    let child_tokens = vec![100_u32, 200, 300, 400];
    let child_persistent_prompt_cache_block_key = parent_persistent_prompt_cache_block_key
        .for_child_block(&child_tokens)
        .expect("the chained persistent prompt-cache block identity should hash a non-empty child block");

    assert_eq!(
        child_persistent_prompt_cache_block_key.token_count(),
        child_tokens.len()
    );
    assert_eq!(child_persistent_prompt_cache_block_key.block_index(), 1);
    assert_ne!(
        child_persistent_prompt_cache_block_key.block_hash(),
        parent_persistent_prompt_cache_block_key.block_hash()
    );
}

#[test]
fn should_reject_an_empty_block_token_sequence() {
    let empty_block_result = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &[],
    );

    assert!(empty_block_result.is_err());
}

#[test]
fn should_reject_a_block_token_sequence_above_the_certified_block_size() {
    let oversized_block_tokens = vec![0_u32; BLOCK_TOKEN_COUNT + 1];
    let oversized_block_result = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &oversized_block_tokens,
    );

    assert!(oversized_block_result.is_err());
}

#[test]
fn should_produce_the_same_hash_for_identical_model_revision_and_token_content() {
    let block_tokens = vec![7_u32, 14, 21, 28];

    let first_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the first identical block should hash");
    let second_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the second identical block should hash");

    assert_eq!(
        first_persistent_prompt_cache_block_key.block_hash(),
        second_persistent_prompt_cache_block_key.block_hash()
    );
}

#[test]
fn should_isolate_blocks_across_different_model_revisions() {
    let block_tokens = vec![7_u32, 14, 21, 28];
    let stale_revision = "0000000000000000000000000000000000000000";

    let certified_persistent_prompt_cache_block_key =
        PersistentPromptCacheBlockKey::for_root_block(
            ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
            ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
            &block_tokens,
        )
        .expect("the certified revision should hash");
    let stale_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        stale_revision,
        &block_tokens,
    )
    .expect("the stale revision should hash");

    assert_ne!(
        certified_persistent_prompt_cache_block_key.block_hash(),
        stale_persistent_prompt_cache_block_key.block_hash()
    );
}

#[test]
fn should_isolate_blocks_across_different_model_ids() {
    let block_tokens = vec![7_u32, 14, 21, 28];
    let foreign_model_id = "mlx-community/Ornith-1.0-35B-OptiQ-8bit";

    let certified_persistent_prompt_cache_block_key =
        PersistentPromptCacheBlockKey::for_root_block(
            ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
            ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
            &block_tokens,
        )
        .expect("the certified model should hash");
    let foreign_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        foreign_model_id,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the foreign model should hash");

    assert_ne!(
        certified_persistent_prompt_cache_block_key.block_hash(),
        foreign_persistent_prompt_cache_block_key.block_hash()
    );
}

#[test]
fn should_isolate_root_blocks_across_ordered_image_digests() {
    let block_tokens = vec![7_u32, 14, 21, 28];
    let first_image_digest = [1_u8; 32];
    let second_image_digest = [2_u8; 32];
    let first_image_order_root_key =
        PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
            ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
            ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
            &block_tokens,
            &[first_image_digest, second_image_digest],
        )
        .expect("the first ordered image identity should hash");
    let reversed_image_order_root_key =
        PersistentPromptCacheBlockKey::for_root_block_with_image_digests(
            ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
            ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
            &block_tokens,
            &[second_image_digest, first_image_digest],
        )
        .expect("the reversed ordered image identity should hash");
    let text_only_root_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the text-only identity should hash");

    assert_ne!(
        first_image_order_root_key.block_hash(),
        reversed_image_order_root_key.block_hash()
    );
    assert_ne!(
        first_image_order_root_key.block_hash(),
        text_only_root_key.block_hash()
    );
}

#[test]
fn should_not_reuse_format_five_root_hash_after_domain_seed_rename() {
    let block_tokens = vec![7_u32, 14, 21, 28];
    let current_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &block_tokens,
    )
    .expect("the current persistent prompt-cache block should hash");

    assert_ne!(
        current_persistent_prompt_cache_block_key.block_hash(),
        format_five_root_block_hash(&block_tokens),
        "domain seed rename must move persistent state into a new hash namespace"
    );
}

#[test]
fn should_reject_an_oversized_child_block_even_when_the_parent_is_valid() {
    let parent_persistent_prompt_cache_block_key = PersistentPromptCacheBlockKey::for_root_block(
        ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID,
        ORNITH_1_0_35B_OPTIQ_4BIT_REVISION,
        &[1_u32, 2, 3, 4],
    )
    .expect("the parent persistent prompt-cache block identity should hash");
    let oversized_child_tokens = vec![0_u32; BLOCK_TOKEN_COUNT + 1];

    let oversized_child_result =
        parent_persistent_prompt_cache_block_key.for_child_block(&oversized_child_tokens);

    assert!(oversized_child_result.is_err());
}

fn root_sentinel_hash() -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"astronomical-decoder-cache-root");
    digest.finalize().into()
}

fn format_five_root_block_hash(block_tokens: &[u32]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"astronomical-ornith-cache-root");
    digest.update(b"5");
    digest.update(ORNITH_1_0_35B_OPTIQ_4BIT_MODEL_ID.as_bytes());
    digest.update(ORNITH_1_0_35B_OPTIQ_4BIT_REVISION.as_bytes());
    for block_token in block_tokens {
        digest.update(block_token.to_be_bytes());
    }
    digest.finalize().into()
}
