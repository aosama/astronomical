use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, RequestId, WorkerEvent,
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence, decode_event, encode_event,
};

#[test]
fn should_round_trip_bounded_persistent_prompt_cache_request_diagnostics_on_completion() {
    let completed_event = WorkerEvent::Completed {
        request_id: RequestId::new(42),
        prompt_token_count: 40_001,
        generated_token_count: 12,
        reasoning_token_count: 3,
        cached_token_count: 38_912,
        persistent_prompt_cache_diagnostics: Some(WorkerPersistentPromptCacheRequestDiagnostics {
            lookup_outcome: WorkerPersistentPromptCacheLookupOutcome::Miss,
            block_token_count: 2_048,
            complete_prompt_block_count: 19,
            maximum_restorable_block_count: 19,
            matched_sequence_state_block_count: 0,
            restored_block_count: 0,
            first_missing_sequence_state_block_index: Some(0),
            miss_reason: Some(WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing),
            expected_block_hash_prefix: Some(
                WorkerPersistentPromptCacheExpectedBlockHashPrefix::from_block_hash([1_u8; 32]),
            ),
            startup_cleanup_evidence: Some(startup_cleanup_evidence()),
            published_block_count: 19,
            allocator_bytes_cleared_for_publication: 4_096,
            expert_bytes_reclaimed_for_publication: 8_192,
        }),
        reason: ChatGenerationCompletionReason::EndOfSequence,
    };

    let serialized_event =
        encode_event(&completed_event).expect("the completed event diagnostics should serialize");
    let decoded_event = decode_event(&serialized_event)
        .expect("the completed event diagnostics should deserialize");

    assert_eq!(decoded_event, completed_event);
}

#[test]
fn should_reject_unknown_startup_cleanup_evidence_fields() {
    let serialized_diagnostics = serde_json::json!({
        "lookup_outcome": "miss",
        "block_token_count": 2048,
        "complete_prompt_block_count": 1,
        "maximum_restorable_block_count": 1,
        "matched_sequence_state_block_count": 0,
        "restored_block_count": 0,
        "first_missing_sequence_state_block_index": 0,
        "miss_reason": "root_sequence_state_block_missing",
        "expected_block_hash_prefix": "0101010101010101",
        "startup_cleanup_evidence": {
            "interrupted_transaction_recovery": {"artifact_count": 1, "block_count": 0, "byte_count": 64},
            "obsolete_format": {"artifact_count": 1, "block_count": 0, "byte_count": 128},
            "corrupt_current_format": {"artifact_count": 0, "block_count": 1, "byte_count": 256},
            "quota_eviction": {"artifact_count": 0, "block_count": 2, "byte_count": 512},
            "unexpected_local_detail": "/fictional/private/cache"
        },
        "published_block_count": 0,
        "allocator_bytes_cleared_for_publication": 0,
        "expert_bytes_reclaimed_for_publication": 0
    });

    let deserialization_error = serde_json::from_value::<
        WorkerPersistentPromptCacheRequestDiagnostics,
    >(serialized_diagnostics)
    .expect_err("unknown cleanup evidence must be rejected");

    assert!(
        deserialization_error
            .to_string()
            .contains("unexpected_local_detail")
    );
}

#[test]
fn should_round_trip_null_startup_cleanup_evidence() {
    let serialized_diagnostics = serde_json::json!({
        "lookup_outcome": "miss",
        "block_token_count": 2048,
        "complete_prompt_block_count": 0,
        "maximum_restorable_block_count": 0,
        "matched_sequence_state_block_count": 0,
        "restored_block_count": 0,
        "first_missing_sequence_state_block_index": null,
        "miss_reason": "prompt_too_short_for_persistent_prompt_cache",
        "expected_block_hash_prefix": null,
        "startup_cleanup_evidence": null,
        "published_block_count": 0,
        "allocator_bytes_cleared_for_publication": 0,
        "expert_bytes_reclaimed_for_publication": 0
    });

    let diagnostics: WorkerPersistentPromptCacheRequestDiagnostics =
        serde_json::from_value(serialized_diagnostics).expect("null evidence should deserialize");

    assert_eq!(diagnostics.startup_cleanup_evidence, None);
}

fn startup_cleanup_evidence() -> WorkerPersistentPromptCacheStartupCleanupEvidence {
    WorkerPersistentPromptCacheStartupCleanupEvidence {
        interrupted_transaction_recovery: WorkerPersistentPromptCacheStartupCleanupCategory {
            artifact_count: u64::MAX,
            block_count: 1,
            byte_count: u64::MAX,
        },
        obsolete_format: WorkerPersistentPromptCacheStartupCleanupCategory {
            artifact_count: 2,
            block_count: 0,
            byte_count: 4_096,
        },
        corrupt_current_format: WorkerPersistentPromptCacheStartupCleanupCategory {
            artifact_count: 0,
            block_count: 3,
            byte_count: 8_192,
        },
        quota_eviction: WorkerPersistentPromptCacheStartupCleanupCategory {
            artifact_count: 1,
            block_count: 4,
            byte_count: 16_384,
        },
    }
}
