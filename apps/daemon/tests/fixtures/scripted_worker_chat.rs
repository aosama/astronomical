use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationOutput, ProtocolWriter, RequestId, WorkerEvent,
    WorkerPersistentPromptCacheExpectedBlockHashPrefix, WorkerPersistentPromptCacheLookupOutcome,
    WorkerPersistentPromptCacheMissReason, WorkerPersistentPromptCacheRequestDiagnostics,
    WorkerPersistentPromptCacheStartupCleanupCategory,
    WorkerPersistentPromptCacheStartupCleanupEvidence,
};

/// One generation's expert-streaming stats for the session-accumulation
/// fixture: the worker's engine-load counters as of this request (they
/// accumulate), plus this request's own prompt/generated token counts (they
/// reset). The split is the contract the daemon-side accumulation depends on.
pub(super) fn expert_streaming_session_stats_snapshot(
    session_request_index: u64,
) -> astronomical_ipc_protocol::WorkerExpertStreamingStats {
    use astronomical_ipc_protocol::{
        ExpertMemoryMode, WorkerExpertResidencySnapshot, WorkerExpertStreamingStats,
    };
    WorkerExpertStreamingStats {
        model_id: Some(String::from(
            "astronomical/expert-streaming-session-fixture",
        )),
        expert_memory_mode: ExpertMemoryMode::Hybrid,
        expert_residency: WorkerExpertResidencySnapshot {
            total_layer_count: 8,
            resident_expert_count: 512,
            resident_expert_payload_bytes: 2_000_000_000,
            complete_expert_payload_byte_count: 8_000_000_000,
            expert_capacity: 256,
            layer_seating: astronomical_ipc_protocol::ExpertLayerSeatingCounts::default(),
        },
        layer_topology: Vec::new(),
        retained_route_assignment_count: 800 * session_request_index,
        storage_route_assignment_count: 200 * session_request_index,
        hot_route_fully_covered_layer_token_count: 70 * session_request_index,
        hot_route_partially_covered_layer_token_count: 20 * session_request_index,
        hot_route_missed_layer_token_count: 10 * session_request_index,
        expert_source_read_call_count: 240 * session_request_index,
        expert_source_read_byte_count: 4_000_000_000 * session_request_index,
        expert_source_read_elapsed_nanoseconds: 3_000_000_000 * session_request_index,
        prefill_source_read_elapsed_nanoseconds: 2_000_000_000 * session_request_index,
        decode_source_read_elapsed_nanoseconds: 1_000_000_000 * session_request_index,
        expert_source_physical_disk_read_bytes: Some(3_000_000_000 * session_request_index),
        avoided_complete_layer_source_bytes: 6_000_000_000 * session_request_index,
        mandatory_prefill_source_bytes: 10_000_000_000 * session_request_index,
        mandatory_decode_source_bytes: 2_000_000_000 * session_request_index,
        prompt_token_count: 100,
        generated_token_count: 10 * session_request_index,
    }
}

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
                expert_residency: None,
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

pub(super) async fn send_expert_streaming_session_chat<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
    session_request_index: u64,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    // A real worker pushes one ExpertStreamingStats event at each generation
    // finalization: engine-load counters (accumulated) plus the finished
    // request's own token counts (reset per reply).
    event_writer
        .send_event(&WorkerEvent::ExpertStreamingStats {
            stats: expert_streaming_session_stats_snapshot(session_request_index),
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: "session totals accumulate".to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 100,
            generated_token_count: 10,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        })
        .await
}

pub(super) async fn send_structured_json_reply<WriteTransport>(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<WriteTransport>,
) -> Result<(), astronomical_ipc_protocol::ProtocolError>
where
    WriteTransport: tokio::io::AsyncWrite + Unpin,
{
    // The scripted fidelity of the root schema journey: the structured
    // generation arm answers with the schema-conforming object so the
    // journey's request and assertions stay identical across fidelities.
    // Grounded in the repository's mandated Romeo and Juliet source.
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: r#"{"speaker":"Juliet","household":"Capulet"}"#.to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::Completed {
            request_id,
            prompt_token_count: 2,
            generated_token_count: 1,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::EndOfSequence,
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
            expert_residency: None,
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
        partial_tail_block_token_count: None,
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
        expert_bytes_reclaimed_for_restore: 0,
        in_memory_prefix_adoption: None,
    }
}
