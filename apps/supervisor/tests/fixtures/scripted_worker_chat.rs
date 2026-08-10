use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationOutput, ProtocolWriter, RequestId, WorkerEvent,
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};

pub(super) async fn send_accepted_chat<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    for (sequence_number, generated_token_count, outputs) in [
        (
            0,
            1,
            vec![ChatGenerationOutput::Reasoning {
                text: "accepted chat reasoning".to_owned(),
            }],
        ),
        (
            1,
            1,
            vec![ChatGenerationOutput::Text {
                text: "accepted chat text".to_owned(),
            }],
        ),
        (
            2,
            3,
            vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 0,
                function_name: "read".to_owned(),
                arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
            }],
        ),
        (
            3,
            4,
            vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 1,
                function_name: "glob".to_owned(),
                arguments_json: r#"{"pattern":"tests/**/*.rs"}"#.to_owned(),
            }],
        ),
    ] {
        event_writer
            .send_event(&WorkerEvent::Output {
                request_id,
                sequence_number,
                generated_token_count,
                outputs,
                mlx_memory_snapshot: None,
            })
            .await?;
    }
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 2,
            generated_token_count: 4,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: Some(accepted_cache_diagnostics()),
            reason: ChatGenerationCompletionReason::ToolCalls,
        })
        .await
}

pub(super) async fn send_simple_completion<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}

pub(super) async fn send_activity_transition<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: "activity transition".to_owned(),
            }],
            mlx_memory_snapshot: None,
        })
        .await?;
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 1,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}

fn accepted_cache_diagnostics() -> WorkerPersistentPromptCacheRequestDiagnostics {
    WorkerPersistentPromptCacheRequestDiagnostics {
        lookup_outcome: WorkerPersistentPromptCacheLookupOutcome::Miss,
        block_token_count: 2_048,
        complete_prompt_block_count: 3,
        maximum_restorable_block_count: 3,
        matched_sequence_state_block_count: 0,
        restored_block_count: 0,
        first_missing_sequence_state_block_index: Some(0),
        miss_reason: Some(WorkerPersistentPromptCacheMissReason::RootSequenceStateBlockMissing),
        expected_block_hash_prefix: Some(
            WorkerPersistentPromptCacheExpectedBlockHashPrefix::from_block_hash([7_u8; 32]),
        ),
        startup_cleanup_evidence: Some(WorkerPersistentPromptCacheStartupCleanupEvidence {
            interrupted_transaction_recovery: WorkerPersistentPromptCacheStartupCleanupCategory {
                artifact_count: 1,
                block_count: 0,
                byte_count: 128,
            },
            obsolete_format: WorkerPersistentPromptCacheStartupCleanupCategory {
                artifact_count: 2,
                block_count: 0,
                byte_count: 256,
            },
            corrupt_current_format: WorkerPersistentPromptCacheStartupCleanupCategory {
                artifact_count: 0,
                block_count: 1,
                byte_count: 512,
            },
            quota_eviction: WorkerPersistentPromptCacheStartupCleanupCategory {
                artifact_count: 0,
                block_count: 0,
                byte_count: 0,
            },
        }),
        published_block_count: 1,
        allocator_bytes_cleared_for_publication: 512,
        expert_bytes_reclaimed_for_publication: 1_024,
    }
}
