use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, RequestId, WorkerEvent,
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    decode_event, encode_event,
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
