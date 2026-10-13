use astronomical_model_serving::{
    PersistentPromptCacheBlockKey, PersistentPromptCachePrefixLookup,
};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

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
                &persistent_prompt_cache_model_contract(),
                &prompt_tokens[block_start..block_end],
            )
            .expect("the test should hash the root block"),
        };
        parent_persistent_prompt_cache_block_key = Some(persistent_prompt_cache_block_key.clone());
        block_keys.push(persistent_prompt_cache_block_key);
    }
    block_keys
}

fn persistent_prompt_cache_block_token_count() -> usize {
    persistent_prompt_cache_model_contract().block_token_count()
}

#[test]
fn should_adopt_the_retained_complete_block_prefix_for_an_extending_prompt() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(4, 7);
    let prompt_block_keys = persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 4);
    // The retained live state covers the first two complete blocks only.
    let retained_block_keys = prompt_block_keys[..2].to_vec();

    let lookup_result = PersistentPromptCachePrefixLookup::for_retained_complete_block_keys(
        &persistent_prompt_cache_model_contract(),
        &prompt_tokens,
        &[],
        &retained_block_keys,
        |_block_hash| true,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count() * 2,
        "adoption must cover exactly the retained complete blocks"
    );
    assert_eq!(
        lookup_result.remaining_tokens().len(),
        prompt_tokens.len() - 2 * persistent_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result
            .last_restored_persistent_prompt_cache_block_key()
            .expect("the adopted chain tip should seed the capture parent")
            .block_hash(),
        retained_block_keys[1].block_hash()
    );
    assert!(
        lookup_result.restored_partial_tail_block_key().is_none(),
        "retained adoption covers complete blocks only"
    );
}

#[test]
fn should_adopt_nothing_when_the_retained_chain_diverges_at_the_root() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 0);
    let divergent_prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 5)
        .iter()
        .map(|token| token + 1_000)
        .collect::<Vec<_>>();
    let retained_block_keys =
        persistent_prompt_cache_block_keys_for_prompt(&divergent_prompt_tokens, 3);

    let lookup_result = PersistentPromptCachePrefixLookup::for_retained_complete_block_keys(
        &persistent_prompt_cache_model_contract(),
        &prompt_tokens,
        &[],
        &retained_block_keys,
        |_block_hash| true,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        0,
        "a different conversation's retained chain must adopt nothing"
    );
}

#[test]
fn should_adopt_a_partial_retained_prefix_when_the_prompt_is_shorter() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(1, 3);
    let longer_prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 0);
    let retained_block_keys =
        persistent_prompt_cache_block_keys_for_prompt(&longer_prompt_tokens, 3);
    // The retained chain's first block must match the short prompt's first
    // block: rebuild the retained keys from the short prompt's chain for the
    // first block and continue the retained chain beyond it.
    let mut matching_retained_block_keys =
        persistent_prompt_cache_block_keys_for_prompt(&prompt_tokens, 1);
    matching_retained_block_keys.extend_from_slice(&retained_block_keys[1..]);

    let lookup_result = PersistentPromptCachePrefixLookup::for_retained_complete_block_keys(
        &persistent_prompt_cache_model_contract(),
        &prompt_tokens,
        &[],
        &matching_retained_block_keys,
        |_block_hash| true,
    );

    // One complete block of a one-block-plus-trailing prompt: the exact-boundary
    // rule keeps the final block for the forward, so adoption covers zero
    // complete blocks here only when the prompt is a single block. With three
    // trailing tokens the final complete block is still restorable.
    assert_eq!(
        lookup_result.restored_token_count(),
        persistent_prompt_cache_block_token_count(),
        "the short prompt adopts its single complete block, keeping the trailing tokens to prefill"
    );
}
