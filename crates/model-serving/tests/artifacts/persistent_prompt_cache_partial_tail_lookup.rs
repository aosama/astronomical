use astronomical_model_serving::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheModelContract,
    PersistentPromptCachePrefixLookup,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

#[test]
fn should_restore_a_partial_tail_block_on_top_of_a_matched_chain_tip() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let chain_tip_block_hash = block_keys[1].block_hash();
    let tail_block_key = chain_tip_block_key(&block_keys[1], &prompt_tokens, 99);

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |block_hash| {
                block_keys
                    .iter()
                    .any(|block_key| block_key.block_hash() == *block_hash)
                    || *block_hash == tail_block_key.block_hash()
            },
            |block_hash| {
                *block_hash == chain_tip_block_hash || *block_hash == tail_block_key.block_hash()
            },
        );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count() * 2 + 99
    );
    assert_eq!(lookup_result.remaining_tokens().len(), 1);
    assert_eq!(
        lookup_result
            .last_restored_persistent_prompt_cache_block_key()
            .expect("the chain tip should remain the capture parent")
            .block_hash(),
        chain_tip_block_hash
    );
    let restored_tail_block_key = lookup_result
        .restored_partial_tail_block_key()
        .expect("the stored tail should restore");
    assert_eq!(
        restored_tail_block_key.block_hash(),
        tail_block_key.block_hash()
    );
    assert_eq!(restored_tail_block_key.token_count(), 99);
    assert_eq!(
        lookup_result
            .diagnostics()
            .restored_partial_tail_block_token_count(),
        Some(99)
    );
}

#[test]
fn should_prefer_the_longest_stored_tail_under_one_parent() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let chain_tip_block_hash = block_keys[1].block_hash();
    let longest_tail_block_key = chain_tip_block_key(&block_keys[1], &prompt_tokens, 99);
    let shorter_tail_block_key = chain_tip_block_key(&block_keys[1], &prompt_tokens, 50);

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |block_hash| {
                block_keys
                    .iter()
                    .any(|block_key| block_key.block_hash() == *block_hash)
                    || *block_hash == longest_tail_block_key.block_hash()
                    || *block_hash == shorter_tail_block_key.block_hash()
            },
            |block_hash| {
                *block_hash == chain_tip_block_hash
                    || *block_hash == longest_tail_block_key.block_hash()
                    || *block_hash == shorter_tail_block_key.block_hash()
            },
        );

    assert_eq!(
        lookup_result
            .restored_partial_tail_block_key()
            .expect("the longest stored tail should win")
            .token_count(),
        99
    );
}

#[test]
fn should_skip_the_tail_probe_when_no_tail_files_exist() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let chain_tip_block_hash = block_keys[1].block_hash();

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |block_hash| {
                block_keys
                    .iter()
                    .any(|block_key| block_key.block_hash() == *block_hash)
            },
            |block_hash| *block_hash == chain_tip_block_hash,
        );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count() * 2
    );
    assert_eq!(lookup_result.remaining_tokens().len(), 100);
    assert!(lookup_result.restored_partial_tail_block_key().is_none());
    assert_eq!(
        lookup_result
            .diagnostics()
            .restored_partial_tail_block_token_count(),
        None
    );
}

#[test]
fn should_restore_a_root_tail_for_a_prompt_shorter_than_one_block() {
    let prompt_token_count = persistent_prompt_cache_block_token_count() / 2;
    let prompt_tokens: Vec<u32> = (0..prompt_token_count as u32).collect();
    let tail_block_key = PersistentPromptCacheBlockKey::for_root_block(
        persistent_prompt_cache_model_contract_ref(),
        &prompt_tokens[..prompt_token_count - 1],
    )
    .expect("the root tail should hash");

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |block_hash| *block_hash == tail_block_key.block_hash(),
            |block_hash| *block_hash == tail_block_key.block_hash(),
        );

    assert_eq!(lookup_result.restored_token_count(), prompt_token_count - 1);
    assert_eq!(lookup_result.remaining_tokens().len(), 1);
    assert!(
        lookup_result
            .last_restored_persistent_prompt_cache_block_key()
            .is_none()
    );
    assert_eq!(
        lookup_result
            .restored_partial_tail_block_key()
            .expect("the root tail should restore")
            .token_count(),
        prompt_token_count - 1
    );
}

#[test]
fn should_not_probe_a_tail_when_the_newest_snapshot_sits_below_the_chain_tip() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let first_block_hash = block_keys[0].block_hash();

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |block_hash| {
                block_keys
                    .iter()
                    .any(|block_key| block_key.block_hash() == *block_hash)
            },
            |block_hash| *block_hash == first_block_hash,
        );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count()
    );
    assert!(lookup_result.restored_partial_tail_block_key().is_none());
}

#[test]
fn should_not_probe_a_tail_when_the_uncached_suffix_still_holds_complete_blocks() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 0);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 1);
    let first_block_hash = block_keys[0].block_hash();

    let lookup_result =
        PersistentPromptCachePrefixLookup::for_prompt_with_partial_tail_block_recovery(
            persistent_prompt_cache_model_contract_ref(),
            &prompt_tokens,
            &[],
            |_block_hash| true,
            |block_hash| *block_hash == first_block_hash,
        );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count()
    );
    assert!(lookup_result.restored_partial_tail_block_key().is_none());
}

#[test]
fn should_keep_existing_lookup_constructors_tail_blind() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let chain_tip_block_hash = block_keys[1].block_hash();
    let tail_block_key = chain_tip_block_key(&block_keys[1], &prompt_tokens, 99);

    let lookup_result = PersistentPromptCachePrefixLookup::for_prompt(
        persistent_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| {
            block_keys
                .iter()
                .any(|block_key| block_key.block_hash() == *block_hash)
                || *block_hash == tail_block_key.block_hash()
        },
        |block_hash| {
            *block_hash == chain_tip_block_hash || *block_hash == tail_block_key.block_hash()
        },
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count() * 2
    );
    assert!(lookup_result.restored_partial_tail_block_key().is_none());
}

fn chain_tip_block_key(
    chain_tip_block_key: &PersistentPromptCacheBlockKey,
    prompt_tokens: &[u32],
    tail_token_count: usize,
) -> PersistentPromptCacheBlockKey {
    let tail_start = prompt_tokens.len() - 100;
    chain_tip_block_key
        .for_child_block(&prompt_tokens[tail_start..tail_start + tail_token_count])
        .expect("the tail block key should hash")
}

fn prompt_tokens_with_complete_blocks_and_trailing_tokens(
    complete_block_count: usize,
    trailing_token_count: usize,
) -> Vec<u32> {
    let prompt_token_count =
        complete_block_count * persistent_prompt_cache_block_token_count() + trailing_token_count;
    (0..prompt_token_count)
        .map(|token_index| token_index as u32)
        .collect()
}

fn persistent_prompt_cache_block_keys_for_prompt(
    prompt_tokens: &[u32],
    requested_block_count: usize,
) -> Vec<PersistentPromptCacheBlockKey> {
    let mut block_keys = Vec::with_capacity(requested_block_count);
    let mut parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey> = None;
    for block_index in 0..requested_block_count {
        let block_start = block_index * persistent_prompt_cache_block_token_count();
        let block_end = block_start + persistent_prompt_cache_block_token_count();
        let persistent_prompt_cache_block_key = match parent_persistent_prompt_cache_block_key {
            Some(ref parent_persistent_prompt_cache_block_key) => {
                parent_persistent_prompt_cache_block_key
                    .for_child_block(&prompt_tokens[block_start..block_end])
                    .expect("the test should hash a child block")
            }
            None => PersistentPromptCacheBlockKey::for_root_block(
                persistent_prompt_cache_model_contract_ref(),
                &prompt_tokens[block_start..block_end],
            )
            .expect("the test should hash the root block"),
        };
        parent_persistent_prompt_cache_block_key = Some(persistent_prompt_cache_block_key.clone());
        block_keys.push(persistent_prompt_cache_block_key);
    }
    block_keys
}

fn persistent_prompt_cache_model_contract_ref() -> &'static PersistentPromptCacheModelContract {
    static PERSISTENT_PROMPT_CACHE_MODEL_CONTRACT: std::sync::OnceLock<
        PersistentPromptCacheModelContract,
    > = std::sync::OnceLock::new();
    PERSISTENT_PROMPT_CACHE_MODEL_CONTRACT.get_or_init(persistent_prompt_cache_model_contract)
}

fn persistent_prompt_cache_block_token_count() -> usize {
    persistent_prompt_cache_model_contract_ref().block_token_count()
}
