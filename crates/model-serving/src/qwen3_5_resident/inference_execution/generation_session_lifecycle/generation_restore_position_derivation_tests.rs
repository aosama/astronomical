use astronomical_ipc_protocol::WorkerPersistentPromptCacheLookupOutcome;
use astronomical_ipc_protocol::WorkerPersistentPromptCacheRequestDiagnostics;

use super::super::decoder_state_reuse::PersistentPromptCacheRestoreOutcome;
use super::derive_generation_session_position;

fn zeroed_request_diagnostics() -> WorkerPersistentPromptCacheRequestDiagnostics {
    WorkerPersistentPromptCacheRequestDiagnostics {
        lookup_outcome: WorkerPersistentPromptCacheLookupOutcome::Hit,
        block_token_count: 0,
        complete_prompt_block_count: 0,
        maximum_restorable_block_count: 0,
        matched_sequence_state_block_count: 0,
        restored_block_count: 0,
        partial_tail_block_token_count: None,
        first_missing_sequence_state_block_index: None,
        miss_reason: None,
        expected_block_hash_prefix: None,
        startup_cleanup_evidence: None,
        published_block_count: 0,
        allocator_bytes_cleared_for_publication: 0,
        expert_bytes_reclaimed_for_publication: 0,
        expert_bytes_reclaimed_for_restore: 0,
        in_memory_prefix_adoption: None,
    }
}

fn restore_outcome(
    persistent_prompt_cache_token_count: u32,
    restored_token_count: usize,
) -> PersistentPromptCacheRestoreOutcome {
    PersistentPromptCacheRestoreOutcome {
        persistent_prompt_cache_token_count,
        restored_token_count,
        last_restored_persistent_prompt_cache_block_key: None,
        persistent_prompt_cache_diagnostics: zeroed_request_diagnostics(),
    }
}

#[test]
fn should_map_every_restored_field_into_its_session_position_slot() {
    let session_position = derive_generation_session_position(Some(restore_outcome(64, 64)));

    assert_eq!(session_position.persistent_prompt_cache_token_count, 64);
    assert_eq!(session_position.prefill_cursor, 64);
    assert_eq!(session_position.next_position_tokens, 64);
    assert_eq!(session_position.restored_target_work_token_count, 64);
    assert!(
        session_position
            .persistent_prompt_cache_diagnostics
            .is_some()
    );
}

#[test]
fn should_keep_the_cursor_and_next_position_distinct_when_a_partial_tail_block_restores() {
    // A partial tail block restores more tokens than the complete-block
    // token count reports: the cursor advances by the restored tokens while
    // positions continue from the persistent prompt-cache token count.
    let session_position = derive_generation_session_position(Some(restore_outcome(64, 100)));

    assert_eq!(session_position.prefill_cursor, 100);
    assert_eq!(session_position.next_position_tokens, 64);
}

#[test]
fn should_reset_the_whole_session_position_when_nothing_was_restored() {
    let session_position = derive_generation_session_position(None);

    assert_eq!(session_position.persistent_prompt_cache_token_count, 0);
    assert_eq!(session_position.prefill_cursor, 0);
    assert_eq!(session_position.next_position_tokens, 0);
    assert_eq!(session_position.restored_target_work_token_count, 0);
    assert_eq!(
        session_position.last_restored_persistent_prompt_cache_block_key,
        None
    );
    assert!(
        session_position
            .persistent_prompt_cache_diagnostics
            .is_none()
    );
}
