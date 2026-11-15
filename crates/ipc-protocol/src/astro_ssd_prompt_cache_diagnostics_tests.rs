use crate::{
    ChatGenerationCompletionReason, RequestId, WorkerAstroSsdPromptCacheExpectedBlockHashPrefix,
    WorkerAstroSsdPromptCacheLookupOutcome, WorkerAstroSsdPromptCacheMissReason,
    WorkerAstroSsdPromptCacheRequestDiagnostics, WorkerAstroSsdPromptCacheStartupCleanupCategory,
    WorkerAstroSsdPromptCacheStartupCleanupEvidence, WorkerEvent, decode_event, encode_event,
};

#[test]
fn should_round_trip_bounded_astro_ssd_prompt_cache_request_diagnostics_on_completion() {
    let completed_event = WorkerEvent::Completed {
        request_id: RequestId::new(42),
        prompt_token_count: 40_001,
        generated_token_count: 12,
        reasoning_token_count: 3,
        astro_ssd_prompt_cache_token_count: 38_912,
        astro_ssd_prompt_cache_diagnostics: Some(WorkerAstroSsdPromptCacheRequestDiagnostics {
            lookup_outcome: WorkerAstroSsdPromptCacheLookupOutcome::Miss,
            block_token_count: 2_048,
            complete_prompt_block_count: 19,
            maximum_restorable_block_count: 19,
            matched_sequence_state_block_count: 0,
            restored_block_count: 0,
            partial_tail_block_token_count: None,
            first_missing_sequence_state_block_index: Some(0),
            miss_reason: Some(WorkerAstroSsdPromptCacheMissReason::RootSequenceStateBlockMissing),
            expected_block_hash_prefix: Some(
                WorkerAstroSsdPromptCacheExpectedBlockHashPrefix::from_block_hash([1_u8; 32]),
            ),
            startup_cleanup_evidence: Some(startup_cleanup_evidence()),
            published_block_count: 19,
            allocator_bytes_cleared_for_publication: 4_096,
            expert_bytes_reclaimed_for_publication: 8_192,
            expert_bytes_reclaimed_for_restore: 16_384,
            in_memory_prefix_adoption: None,
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
        "expert_bytes_reclaimed_for_publication": 0,
        "expert_bytes_reclaimed_for_restore": 0
    });

    let deserialization_error =
        serde_json::from_value::<WorkerAstroSsdPromptCacheRequestDiagnostics>(
            serialized_diagnostics,
        )
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
        "miss_reason": "prompt_too_short_for_astro_ssd_prompt_cache",
        "expected_block_hash_prefix": null,
        "startup_cleanup_evidence": null,
        "published_block_count": 0,
        "allocator_bytes_cleared_for_publication": 0,
        "expert_bytes_reclaimed_for_publication": 0,
        "expert_bytes_reclaimed_for_restore": 0
    });

    let diagnostics: WorkerAstroSsdPromptCacheRequestDiagnostics =
        serde_json::from_value(serialized_diagnostics).expect("null evidence should deserialize");

    assert_eq!(diagnostics.startup_cleanup_evidence, None);
}

fn startup_cleanup_evidence() -> WorkerAstroSsdPromptCacheStartupCleanupEvidence {
    WorkerAstroSsdPromptCacheStartupCleanupEvidence {
        interrupted_transaction_recovery: WorkerAstroSsdPromptCacheStartupCleanupCategory {
            artifact_count: u64::MAX,
            block_count: 1,
            byte_count: u64::MAX,
        },
        obsolete_format: WorkerAstroSsdPromptCacheStartupCleanupCategory {
            artifact_count: 2,
            block_count: 0,
            byte_count: 4_096,
        },
        corrupt_current_format: WorkerAstroSsdPromptCacheStartupCleanupCategory {
            artifact_count: 0,
            block_count: 3,
            byte_count: 8_192,
        },
        quota_eviction: WorkerAstroSsdPromptCacheStartupCleanupCategory {
            artifact_count: 1,
            block_count: 4,
            byte_count: 16_384,
        },
    }
}

#[test]
fn should_keep_the_first_eight_hash_bytes_as_sixteen_lowercase_digits() {
    let ascending_block_hash: [u8; 32] =
        core::array::from_fn(|byte_index| u8::try_from(byte_index).expect("index fits a byte"));
    assert_eq!(
        WorkerAstroSsdPromptCacheExpectedBlockHashPrefix::from_block_hash(ascending_block_hash)
            .as_str(),
        "0001020304050607"
    );
    assert_eq!(
        WorkerAstroSsdPromptCacheExpectedBlockHashPrefix::from_block_hash([0xab; 32]).as_str(),
        "abababababababab"
    );
}

#[test]
fn should_refuse_a_hash_prefix_that_is_not_sixteen_lowercase_digits() {
    for foreign_prefix in [
        "0123456789ABCDEF",
        "0123456789abcde",
        "0123456789abcdef0",
        "+123456789abcdef",
        "",
    ] {
        let prefix_json = serde_json::to_string(foreign_prefix).expect("a string serializes");
        assert!(
            serde_json::from_str::<WorkerAstroSsdPromptCacheExpectedBlockHashPrefix>(&prefix_json)
                .is_err(),
            "{foreign_prefix}"
        );
    }
}

#[test]
fn should_accept_a_hash_prefix_of_sixteen_lowercase_digits() {
    let decoded_prefix: WorkerAstroSsdPromptCacheExpectedBlockHashPrefix =
        serde_json::from_str("\"0123456789abcdef\"").expect("a valid prefix decodes");
    assert_eq!(decoded_prefix.as_str(), "0123456789abcdef");
}
