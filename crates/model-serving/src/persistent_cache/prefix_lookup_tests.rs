//! Beside-source unit tests for the pure units extracted out of the prefix
//! lookup (`prefix_lookup.rs`): block-boundary counting, the boundary-snapshot
//! miss reason, and the newest-restorable-boundary scan. The end-to-end lookup
//! contracts, including walking back to an earlier snapshot on a real chain,
//! stay in `prefix_lookup/tests`.

use crate::decoder_cache::{
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};

use super::super::prefix_lookup_steps::{
    boundary_snapshot_miss_reason, complete_and_maximum_restorable_block_counts,
    newest_restorable_boundary_entry,
};
use super::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheMissReason,
    PersistentPromptCacheModelContract,
};

const FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT: usize = 128;

#[test]
fn should_count_every_complete_block_as_restorable_when_the_prompt_is_not_boundary_aligned() {
    let (complete_prompt_block_count, maximum_restorable_block_count) =
        complete_and_maximum_restorable_block_counts(2_500, 1_024, false);

    assert_eq!(complete_prompt_block_count, 2);
    assert_eq!(maximum_restorable_block_count, 2);
}

#[test]
fn should_accept_every_complete_block_when_exact_boundary_restore_is_allowed() {
    let (complete_prompt_block_count, maximum_restorable_block_count) =
        complete_and_maximum_restorable_block_counts(2_048, 1_024, true);

    assert_eq!(complete_prompt_block_count, 2);
    assert_eq!(maximum_restorable_block_count, 2);
}

#[test]
fn should_reserve_one_block_for_forward_processing_on_an_exact_block_boundary() {
    let (complete_prompt_block_count, maximum_restorable_block_count) =
        complete_and_maximum_restorable_block_counts(2_048, 1_024, false);

    assert_eq!(complete_prompt_block_count, 2);
    assert_eq!(maximum_restorable_block_count, 1);
}

#[test]
fn should_report_no_restorable_blocks_for_prompts_shorter_than_one_block() {
    let (complete_prompt_block_count, maximum_restorable_block_count) =
        complete_and_maximum_restorable_block_counts(1_023, 1_024, false);

    assert_eq!(complete_prompt_block_count, 0);
    assert_eq!(maximum_restorable_block_count, 0);
}

#[test]
fn should_report_no_restorable_blocks_for_an_empty_prompt() {
    let (complete_prompt_block_count, maximum_restorable_block_count) =
        complete_and_maximum_restorable_block_counts(0, 1_024, true);

    assert_eq!(complete_prompt_block_count, 0);
    assert_eq!(maximum_restorable_block_count, 0);
}

#[test]
fn should_report_a_missing_root_sequence_state_block_only_at_the_root() {
    let miss_reason = boundary_snapshot_miss_reason(true, Some(0));

    assert_eq!(
        miss_reason,
        PersistentPromptCacheMissReason::RootSequenceStateBlockMissing
    );
}

#[test]
fn should_report_a_missing_boundary_snapshot_for_a_gap_after_the_root() {
    let miss_reason = boundary_snapshot_miss_reason(true, Some(3));

    assert_eq!(
        miss_reason,
        PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
    );
}

#[test]
fn should_report_a_missing_boundary_snapshot_when_every_matched_block_is_present() {
    let miss_reason = boundary_snapshot_miss_reason(true, None);

    assert_eq!(
        miss_reason,
        PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
    );
}

#[test]
fn should_report_a_missing_boundary_snapshot_for_boundary_only_models_regardless_of_gaps() {
    let miss_reason_at_gap = boundary_snapshot_miss_reason(false, Some(0));
    let miss_reason_without_gap = boundary_snapshot_miss_reason(false, None);

    assert_eq!(
        miss_reason_at_gap,
        PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
    );
    assert_eq!(
        miss_reason_without_gap,
        PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing
    );
}

fn every_snapshot_exists(_block_hash: &[u8; 32]) -> bool {
    true
}

fn no_snapshot_exists(_block_hash: &[u8; 32]) -> bool {
    false
}

fn synthetic_sequence_only_contract() -> PersistentPromptCacheModelContract {
    // One append-only attention layer with four-token growth derives a
    // four-token block, so fixture chains stay small and readable.
    PersistentPromptCacheModelContract::resolve(
        "prefix-lookup-unit-fixture".to_owned(),
        "revision".to_owned(),
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
            4,
        )])
        .expect("the synthetic sequence-only layout should be valid"),
        FIXTURE_MAXIMUM_CONTEXT_TOKEN_COUNT,
        1_000_000,
        1_000_000,
        None,
        4,
    )
    .expect("the synthetic layout should resolve a storage contract")
}

fn synthetic_block_key_chain(
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    matched_block_count: usize,
) -> Vec<PersistentPromptCacheBlockKey> {
    let persistent_prompt_cache_block_token_count =
        persistent_prompt_cache_model_contract.block_token_count();
    let mut matched_persistent_prompt_cache_block_keys = Vec::new();
    let mut parent_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey> = None;
    for block_index in 0..matched_block_count {
        let block_tokens = vec![block_index as u32; persistent_prompt_cache_block_token_count];
        let persistent_prompt_cache_block_key = match parent_persistent_prompt_cache_block_key {
            None => PersistentPromptCacheBlockKey::for_root_block(
                persistent_prompt_cache_model_contract,
                &block_tokens,
            ),
            Some(parent_persistent_prompt_cache_block_key) => {
                parent_persistent_prompt_cache_block_key.for_child_block(&block_tokens)
            }
        }
        .expect("the synthetic block key should construct");
        matched_persistent_prompt_cache_block_keys.push(persistent_prompt_cache_block_key.clone());
        parent_persistent_prompt_cache_block_key = Some(persistent_prompt_cache_block_key);
    }
    matched_persistent_prompt_cache_block_keys
}

#[test]
fn should_select_the_chain_tip_as_the_newest_restorable_boundary_when_every_snapshot_exists() {
    let persistent_prompt_cache_model_contract = synthetic_sequence_only_contract();
    let matched_persistent_prompt_cache_block_keys =
        synthetic_block_key_chain(&persistent_prompt_cache_model_contract, 3);
    let chain_tip_block_hash = matched_persistent_prompt_cache_block_keys[2].block_hash();

    let boundary_entry = newest_restorable_boundary_entry(
        true,
        &matched_persistent_prompt_cache_block_keys,
        every_snapshot_exists,
    );

    assert_eq!(
        boundary_entry.map(|(recurrent_snapshot_block_index, _)| recurrent_snapshot_block_index),
        Some(2)
    );
    assert_eq!(
        boundary_entry.map(|(_, restored_block_key)| restored_block_key.block_hash()),
        Some(chain_tip_block_hash)
    );
}

#[test]
fn should_return_no_boundary_entry_when_no_matched_block_has_a_snapshot() {
    let persistent_prompt_cache_model_contract = synthetic_sequence_only_contract();
    let matched_persistent_prompt_cache_block_keys =
        synthetic_block_key_chain(&persistent_prompt_cache_model_contract, 3);

    let boundary_entry = newest_restorable_boundary_entry(
        true,
        &matched_persistent_prompt_cache_block_keys,
        no_snapshot_exists,
    );

    assert_eq!(boundary_entry, None);
}

#[test]
fn should_return_the_chain_tip_without_consulting_snapshots_for_boundary_only_models() {
    let persistent_prompt_cache_model_contract = synthetic_sequence_only_contract();
    let matched_persistent_prompt_cache_block_keys =
        synthetic_block_key_chain(&persistent_prompt_cache_model_contract, 3);

    let boundary_entry = newest_restorable_boundary_entry(
        false,
        &matched_persistent_prompt_cache_block_keys,
        no_snapshot_exists,
    );

    assert_eq!(
        boundary_entry.map(|(recurrent_snapshot_block_index, _)| recurrent_snapshot_block_index),
        Some(2)
    );
}

#[test]
fn should_return_no_boundary_entry_for_an_empty_matched_chain() {
    let persistent_prompt_cache_model_contract = synthetic_sequence_only_contract();
    let matched_persistent_prompt_cache_block_keys =
        synthetic_block_key_chain(&persistent_prompt_cache_model_contract, 0);

    let boundary_entry = newest_restorable_boundary_entry(
        true,
        &matched_persistent_prompt_cache_block_keys,
        every_snapshot_exists,
    );

    assert_eq!(boundary_entry, None);
}
