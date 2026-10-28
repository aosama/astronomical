//! Unit coverage for the shared restore support: the IPC diagnostics mapping,
//! the startup-cleanup miss reasons, and the restore block-key derivation —
//! positive, boundary, and negative cases. These rules previously lived as
//! untested copies inside both engines' restore functions.

use astronomical_ipc_protocol::{
    WorkerPersistentPromptCacheLookupOutcome, WorkerPersistentPromptCacheMissReason,
};

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::restore_support::{
    miss_reason_warrants_startup_cleanup_evidence, persistent_prompt_cache_request_diagnostics,
    restored_persistent_prompt_cache_block_key,
};
use crate::persistent_cache::restore_support_tests_support::test_model_contract;
use crate::{
    InferenceEngineError, PersistentPromptCacheLookupDiagnostics, PersistentPromptCacheMissReason,
};

/// A miss-shaped lookup diagnostics record for the mapping tests.
fn miss_lookup_diagnostics() -> PersistentPromptCacheLookupDiagnostics {
    PersistentPromptCacheLookupDiagnostics::for_tests(
        0,
        0,
        0,
        Some(PersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache),
    )
}

/// A hit-shaped lookup diagnostics record with two complete blocks matched.
fn hit_lookup_diagnostics() -> PersistentPromptCacheLookupDiagnostics {
    PersistentPromptCacheLookupDiagnostics::for_tests(2, 3, 2, None)
}

#[test]
fn should_map_a_miss_lookup_into_miss_ipc_diagnostics() {
    let request_diagnostics =
        persistent_prompt_cache_request_diagnostics(64, &miss_lookup_diagnostics(), 0, None);

    assert_eq!(
        request_diagnostics.lookup_outcome,
        WorkerPersistentPromptCacheLookupOutcome::Miss
    );
    assert_eq!(request_diagnostics.block_token_count, 64);
    assert_eq!(request_diagnostics.restored_block_count, 0);
    assert_eq!(
        request_diagnostics.miss_reason,
        Some(WorkerPersistentPromptCacheMissReason::PromptTooShortForPersistentPromptCache)
    );
    assert_eq!(request_diagnostics.expert_bytes_reclaimed_for_restore, 0);
}

#[test]
fn should_map_a_partial_tail_hit_into_hit_diagnostics_with_the_tail_counted() {
    let request_diagnostics =
        persistent_prompt_cache_request_diagnostics(64, &hit_lookup_diagnostics(), 160, Some(32));

    assert_eq!(
        request_diagnostics.lookup_outcome,
        WorkerPersistentPromptCacheLookupOutcome::Hit
    );
    // Two complete blocks plus the partial tail: three restored blocks.
    assert_eq!(request_diagnostics.restored_block_count, 3);
    assert_eq!(request_diagnostics.partial_tail_block_token_count, Some(32));
    assert_eq!(request_diagnostics.complete_prompt_block_count, 2);
    assert_eq!(request_diagnostics.matched_sequence_state_block_count, 2);
}

#[test]
fn should_bound_every_ipc_diagnostic_field_at_u64_max() {
    let request_diagnostics = persistent_prompt_cache_request_diagnostics(
        usize::MAX,
        &hit_lookup_diagnostics(),
        usize::MAX,
        Some(usize::MAX),
    );

    // usize::MAX on a 64-bit host is exactly u64::MAX, so the direct
    // conversions saturate on the bound instead of wrapping. The restored
    // block count, however, is usize arithmetic first: a full-context
    // restore of one block plus the tail divides down to two blocks.
    assert_eq!(request_diagnostics.block_token_count, u64::MAX);
    assert_eq!(request_diagnostics.restored_block_count, 2);
    assert_eq!(
        request_diagnostics.partial_tail_block_token_count,
        Some(u64::MAX)
    );
}

#[test]
fn should_warrant_startup_cleanup_evidence_only_for_curable_miss_reasons() {
    let root_block_missing = PersistentPromptCacheLookupDiagnostics::for_tests(
        0,
        0,
        0,
        Some(PersistentPromptCacheMissReason::RootSequenceStateBlockMissing),
    );
    let boundary_missing = PersistentPromptCacheLookupDiagnostics::for_tests(
        0,
        0,
        0,
        Some(PersistentPromptCacheMissReason::BoundaryStateSnapshotMissing),
    );

    assert!(miss_reason_warrants_startup_cleanup_evidence(
        &root_block_missing
    ));
    assert!(miss_reason_warrants_startup_cleanup_evidence(
        &boundary_missing
    ));
    assert!(!miss_reason_warrants_startup_cleanup_evidence(
        &miss_lookup_diagnostics()
    ));
    assert!(!miss_reason_warrants_startup_cleanup_evidence(
        &hit_lookup_diagnostics()
    ));
}

#[test]
fn should_derive_the_root_block_key_from_the_pinned_model_contract() {
    let persistent_prompt_cache_model_contract = test_model_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let prompt_token_ids: Vec<u32> = (0..block_token_count).map(|index| index as u32).collect();

    let root_block_key = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        0,
        block_token_count,
        0,
        &[],
        None,
        &persistent_prompt_cache_model_contract,
    )
    .expect("the root block key should derive from the model contract");

    assert!(!root_block_key.block_hash().is_empty());
}

#[test]
fn should_chain_child_block_keys_from_their_parent() {
    let persistent_prompt_cache_model_contract = test_model_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let prompt_token_ids: Vec<u32> = (0..block_token_count * 2)
        .map(|index| index as u32 + 7)
        .collect();

    let root_block_key = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        0,
        block_token_count,
        0,
        &[],
        None,
        &persistent_prompt_cache_model_contract,
    )
    .expect("the root block key should derive");
    let child_block_key = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        block_token_count,
        block_token_count * 2,
        1,
        &[],
        Some(&root_block_key),
        &persistent_prompt_cache_model_contract,
    )
    .expect("the child block key should chain from its parent");

    // The child's identity must not collide with a root built from the same
    // tokens: the parent hash participates in the chain.
    assert_ne!(root_block_key.block_hash(), child_block_key.block_hash());
}

#[test]
fn should_reject_a_causal_plan_that_skips_a_restored_block() {
    let persistent_prompt_cache_model_contract = test_model_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let prompt_token_ids: Vec<u32> = (0..block_token_count).map(|index| index as u32).collect();
    let causal_plan_missing_later_blocks = [PersistentPromptCacheBlockCausalInput::empty()];

    let chain_not_covered_error = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        0,
        block_token_count,
        5,
        &causal_plan_missing_later_blocks,
        None,
        &persistent_prompt_cache_model_contract,
    )
    .expect_err("a partial causal plan must stop the restore");
    let InferenceEngineError::Fatal { reason } = chain_not_covered_error else {
        panic!("expected a fatal engine error");
    };
    assert_eq!(
        reason,
        "persistent prompt-cache causal input plan does not cover restored block"
    );
}

#[test]
fn should_reject_a_non_root_restore_without_a_parent_key() {
    let persistent_prompt_cache_model_contract = test_model_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let prompt_token_ids: Vec<u32> = (0..block_token_count).map(|index| index as u32).collect();

    let chain_lost_error = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        block_token_count,
        block_token_count * 2,
        1,
        &[],
        None,
        &persistent_prompt_cache_model_contract,
    )
    .expect_err("a parentless child must stop the restore");
    let InferenceEngineError::Fatal { reason } = chain_lost_error else {
        panic!("expected a fatal engine error");
    };
    assert_eq!(
        reason,
        "persistent prompt-cache block identity chain was lost during restore"
    );
}

#[test]
fn should_keep_the_causal_plan_positive_case_for_block_zero() {
    let persistent_prompt_cache_model_contract = test_model_contract();
    let block_token_count = persistent_prompt_cache_model_contract.block_token_count();
    let prompt_token_ids: Vec<u32> = (0..block_token_count).map(|index| index as u32).collect();
    let causal_plan_covering_block_zero = [PersistentPromptCacheBlockCausalInput::empty()];

    let root_block_key = restored_persistent_prompt_cache_block_key(
        &prompt_token_ids,
        0,
        block_token_count,
        0,
        &causal_plan_covering_block_zero,
        None,
        &persistent_prompt_cache_model_contract,
    )
    .expect("block zero resolves through the plan's first entry");

    assert!(!root_block_key.block_hash().is_empty());
}
