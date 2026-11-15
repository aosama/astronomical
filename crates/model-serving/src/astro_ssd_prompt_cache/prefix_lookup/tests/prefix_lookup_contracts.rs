use crate::{
    AstroSsdPromptCacheBlockCausalInput, AstroSsdPromptCacheBlockKey,
    AstroSsdPromptCacheMissReason, AstroSsdPromptCacheModelContract,
    AstroSsdPromptCachePrefixLookup,
};

use super::lookup_fixtures::astro_ssd_prompt_cache_model_contract;

#[test]
fn should_report_no_astro_ssd_prompt_cache_hit_when_no_complete_blocks_exist() {
    let prompt_tokens: Vec<u32> = (0..1_500).map(|token_index| token_index as u32).collect();
    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |_block_hash| false,
        |_block_hash| false,
    );

    assert_eq!(lookup_result.restored_token_count(), 0);
    assert!(lookup_result.remaining_tokens().eq(&prompt_tokens));
    assert!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .is_none()
    );
}

#[test]
fn should_restore_through_the_chain_tip_when_its_recurrent_snapshot_exists() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let chain_tip_block_hash = block_keys[1].block_hash();

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| {
            block_keys
                .iter()
                .any(|block_key| block_key.block_hash() == *block_hash)
        },
        |block_hash| *block_hash == chain_tip_block_hash,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        astro_ssd_prompt_cache_block_token_count() * 2
    );
    assert_eq!(lookup_result.remaining_tokens().len(), 100);
    assert_eq!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .expect("the chain-tip snapshot should restore through the second block")
            .block_hash(),
        chain_tip_block_hash
    );
}

#[test]
fn should_walk_back_to_the_latest_available_recurrent_snapshot() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 500);
    let block_keys = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 3);
    let first_block_hash = block_keys[0].block_hash();

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| {
            block_keys
                .iter()
                .any(|block_key| block_key.block_hash() == *block_hash)
        },
        |block_hash| *block_hash == first_block_hash,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        astro_ssd_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result.remaining_tokens().len(),
        prompt_tokens.len() - astro_ssd_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .expect("the earlier snapshot should become the restored boundary")
            .block_hash(),
        first_block_hash
    );
    let lookup_diagnostics = lookup_result.diagnostics();
    assert_eq!(lookup_diagnostics.matched_sequence_state_block_count(), 3);
    assert_eq!(
        lookup_diagnostics.newest_boundary_state_snapshot_block_index(),
        Some(0)
    );
    assert_eq!(lookup_diagnostics.miss_reason(), None);
}

#[test]
fn should_return_a_cold_miss_when_kv_blocks_exist_without_any_recurrent_snapshot() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let block_keys = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| {
            block_keys
                .iter()
                .any(|block_key| block_key.block_hash() == *block_hash)
        },
        |_block_hash| false,
    );

    assert_eq!(lookup_result.restored_token_count(), 0);
    assert_eq!(lookup_result.remaining_tokens().len(), prompt_tokens.len());
    assert!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .is_none()
    );
    let lookup_diagnostics = lookup_result.diagnostics();
    assert_eq!(lookup_diagnostics.matched_sequence_state_block_count(), 2);
    assert_eq!(
        lookup_diagnostics.first_missing_sequence_state_block_index(),
        None
    );
    assert_eq!(
        lookup_diagnostics.newest_boundary_state_snapshot_block_index(),
        None
    );
    assert_eq!(
        lookup_diagnostics.miss_reason(),
        Some(AstroSsdPromptCacheMissReason::BoundaryStateSnapshotMissing)
    );
}

#[test]
fn should_keep_the_final_block_for_forward_processing_when_prompt_ends_on_a_block_boundary() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 0);
    let first_block_key = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 1)
        .pop()
        .expect("the test should produce the first block key");
    let first_block_hash = first_block_key.block_hash();

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |_block_hash| true,
        |block_hash| *block_hash == first_block_hash,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        astro_ssd_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result.remaining_tokens().len(),
        astro_ssd_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .expect("one block should restore for a two-block exact-boundary prompt")
            .block_hash(),
        first_block_hash
    );
}

#[test]
fn should_restore_a_complete_exact_boundary_when_the_caller_only_needs_decoder_state() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 0);
    let block_keys = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 2);
    let final_block_hash = block_keys[1].block_hash();

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_complete_prefix(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| {
            block_keys
                .iter()
                .any(|block_key| block_key.block_hash() == *block_hash)
        },
        |block_hash| *block_hash == final_block_hash,
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        astro_ssd_prompt_cache_block_token_count() * 2
    );
    assert!(lookup_result.remaining_tokens().is_empty());
    assert_eq!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .expect("the complete prefix should restore its final block")
            .block_hash(),
        final_block_hash
    );
}

#[test]
fn should_not_match_blocks_when_the_root_block_differs() {
    let original_prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 0);
    let mut modified_prompt_tokens = original_prompt_tokens.clone();
    modified_prompt_tokens[0] = 999_u32;
    let original_root_block_key = AstroSsdPromptCacheBlockKey::for_root_block(
        astro_ssd_prompt_cache_model_contract_ref(),
        &original_prompt_tokens[..astro_ssd_prompt_cache_block_token_count()],
    )
    .expect("the test should hash the original root block");
    let requested_root_block_key = AstroSsdPromptCacheBlockKey::for_root_block(
        astro_ssd_prompt_cache_model_contract_ref(),
        &modified_prompt_tokens[..astro_ssd_prompt_cache_block_token_count()],
    )
    .expect("the test should hash the requested root block");

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &modified_prompt_tokens,
        |block_hash| *block_hash == original_root_block_key.block_hash(),
        |block_hash| *block_hash == original_root_block_key.block_hash(),
    );

    assert_eq!(lookup_result.restored_token_count(), 0);
    assert!(
        lookup_result
            .last_restored_astro_ssd_prompt_cache_block_key()
            .is_none()
    );
    let lookup_diagnostics = lookup_result.diagnostics();
    assert_eq!(lookup_diagnostics.matched_sequence_state_block_count(), 0);
    assert_eq!(
        lookup_diagnostics.first_missing_sequence_state_block_index(),
        Some(0)
    );
    assert_eq!(
        lookup_diagnostics.first_missing_sequence_state_block_hash(),
        Some(requested_root_block_key.block_hash())
    );
    assert_eq!(
        lookup_diagnostics.miss_reason(),
        Some(AstroSsdPromptCacheMissReason::RootSequenceStateBlockMissing)
    );
}

#[test]
fn should_restore_blocks_before_changed_visual_content() {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(2, 100);
    let empty_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
    let cached_visual_input = AstroSsdPromptCacheBlockCausalInput::from_canonical_bytes(&[1]);
    let requested_visual_input = AstroSsdPromptCacheBlockCausalInput::from_canonical_bytes(&[2]);
    let cached_root_block_key = AstroSsdPromptCacheBlockKey::for_root_block_with_causal_input(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens[..astro_ssd_prompt_cache_block_token_count()],
        &empty_causal_input,
    )
    .expect("the root block should hash");
    let cached_child_block_key = cached_root_block_key
        .for_child_block_with_causal_input(
            &prompt_tokens[astro_ssd_prompt_cache_block_token_count()
                ..astro_ssd_prompt_cache_block_token_count() * 2],
            &cached_visual_input,
        )
        .expect("the visual child block should hash");

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt_with_block_causal_inputs(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        &[empty_causal_input, requested_visual_input],
        |block_hash| {
            *block_hash == cached_root_block_key.block_hash()
                || *block_hash == cached_child_block_key.block_hash()
        },
        |block_hash| *block_hash == cached_root_block_key.block_hash(),
    );

    assert_eq!(
        lookup_result.restored_token_count(),
        astro_ssd_prompt_cache_block_token_count()
    );
    assert_eq!(
        lookup_result
            .diagnostics()
            .first_missing_sequence_state_block_index(),
        Some(1)
    );
    assert_eq!(lookup_result.diagnostics().miss_reason(), None);
}

#[test]
fn should_report_missing_recurrent_snapshot_before_missing_child_when_matched_prefix_has_no_snapshot()
 {
    let prompt_tokens = prompt_tokens_with_complete_blocks_and_trailing_tokens(3, 100);
    let block_keys = astro_ssd_prompt_cache_block_keys_for_prompt(&prompt_tokens, 3);
    let first_block_hash = block_keys[0].block_hash();

    let lookup_result = AstroSsdPromptCachePrefixLookup::for_prompt(
        astro_ssd_prompt_cache_model_contract_ref(),
        &prompt_tokens,
        |block_hash| *block_hash == first_block_hash,
        |_block_hash| false,
    );

    assert_eq!(lookup_result.restored_token_count(), 0);
    let lookup_diagnostics = lookup_result.diagnostics();
    assert_eq!(lookup_diagnostics.complete_prompt_block_count(), 3);
    assert_eq!(lookup_diagnostics.maximum_restorable_block_count(), 3);
    assert_eq!(lookup_diagnostics.matched_sequence_state_block_count(), 1);
    assert_eq!(
        lookup_diagnostics.first_missing_sequence_state_block_index(),
        Some(1)
    );
    assert_eq!(
        lookup_diagnostics.first_missing_sequence_state_block_hash(),
        Some(block_keys[1].block_hash())
    );
    assert_eq!(
        lookup_diagnostics.newest_boundary_state_snapshot_block_index(),
        None
    );
    assert_eq!(
        lookup_diagnostics.miss_reason(),
        Some(AstroSsdPromptCacheMissReason::BoundaryStateSnapshotMissing)
    );
}

fn prompt_tokens_with_complete_blocks_and_trailing_tokens(
    complete_block_count: usize,
    trailing_token_count: usize,
) -> Vec<u32> {
    let prompt_token_count =
        complete_block_count * astro_ssd_prompt_cache_block_token_count() + trailing_token_count;
    (0..prompt_token_count)
        .map(|token_index| token_index as u32)
        .collect()
}

fn astro_ssd_prompt_cache_block_keys_for_prompt(
    prompt_tokens: &[u32],
    requested_block_count: usize,
) -> Vec<AstroSsdPromptCacheBlockKey> {
    let mut block_keys = Vec::with_capacity(requested_block_count);
    let mut parent_astro_ssd_prompt_cache_block_key: Option<AstroSsdPromptCacheBlockKey> = None;
    for block_index in 0..requested_block_count {
        let block_start = block_index * astro_ssd_prompt_cache_block_token_count();
        let block_end = block_start + astro_ssd_prompt_cache_block_token_count();
        let astro_ssd_prompt_cache_block_key = match parent_astro_ssd_prompt_cache_block_key {
            Some(ref parent_astro_ssd_prompt_cache_block_key) => {
                parent_astro_ssd_prompt_cache_block_key
                    .for_child_block(&prompt_tokens[block_start..block_end])
                    .expect("the test should hash a child block")
            }
            None => AstroSsdPromptCacheBlockKey::for_root_block(
                astro_ssd_prompt_cache_model_contract_ref(),
                &prompt_tokens[block_start..block_end],
            )
            .expect("the test should hash the root block"),
        };
        parent_astro_ssd_prompt_cache_block_key = Some(astro_ssd_prompt_cache_block_key.clone());
        block_keys.push(astro_ssd_prompt_cache_block_key);
    }
    block_keys
}

fn astro_ssd_prompt_cache_model_contract_ref() -> &'static AstroSsdPromptCacheModelContract {
    static ASTRO_SSD_PROMPT_CACHE_MODEL_CONTRACT: std::sync::OnceLock<
        AstroSsdPromptCacheModelContract,
    > = std::sync::OnceLock::new();
    ASTRO_SSD_PROMPT_CACHE_MODEL_CONTRACT.get_or_init(astro_ssd_prompt_cache_model_contract)
}

fn astro_ssd_prompt_cache_block_token_count() -> usize {
    astro_ssd_prompt_cache_model_contract_ref().block_token_count()
}
