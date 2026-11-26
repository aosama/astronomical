//! Chat fixtures that answer one request with a fixed event sequence and
//! touch no cancellation state. They live apart from the dispatcher in
//! `scripted_worker.rs` so each scripted reply reads on its own.

use std::{error::Error, time::Duration};

use ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationOutput,
    ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolWriter, RequestId, WorkerEvent,
    WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase,
};

use super::scripted_worker_chat;

type FixtureError = Box<dyn Error + Send + Sync>;

/// Answers `model_id` with its scripted event sequence, or with a plain
/// completion for a model no fixture here scripts.
pub(super) async fn send_scripted_chat_output(
    model_id: &str,
    request_id: RequestId,
    maximum_output_tokens: u16,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    match model_id {
        "astronomical/malformed-output-fixture" => {
            send_malformed_output_failure(request_id, event_writer).await?;
        }
        "astronomical/delayed-malformed-output-fixture" => {
            send_progress_then_malformed_output_failure(
                request_id,
                maximum_output_tokens,
                event_writer,
            )
            .await?;
        }
        "astronomical/activity-transition-fixture" => {
            scripted_worker_chat::send_activity_transition(request_id, event_writer).await?;
        }
        "astronomical/duplicate-generation-preparation-fixture" => {
            send_duplicate_generation_preparation(request_id, event_writer).await?;
        }
        "astronomical/out-of-order-chat-fixture" => {
            send_out_of_order_output(request_id, event_writer).await?;
        }
        "astronomical/empty-output-batch-fixture" => {
            send_empty_output_batch(request_id, event_writer).await?;
        }
        "astronomical/invalid-output-batch-fixture" => {
            send_invalid_output_batch(request_id, event_writer).await?;
        }
        "astronomical/over-budget-tool-completion-fixture" => {
            send_over_budget_tool_completion(request_id, event_writer).await?;
        }
        "astronomical/unsolicited-cancellation-fixture" => {
            scripted_worker_chat::send_completion(
                request_id,
                1,
                0,
                ChatGenerationCompletionReason::Cancelled,
                event_writer,
            )
            .await?;
        }
        "astronomical/backpressure-fixture" => {
            send_backpressure_burst(request_id, event_writer).await?;
        }
        "astronomical/prefill-progress-fixture" => {
            send_prefill_progress_reply(request_id, event_writer).await?;
        }
        _ => scripted_worker_chat::send_simple_completion(request_id, &mut *event_writer).await?,
    }
    Ok(())
}

async fn send_malformed_output_failure(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::Failed {
            request_id,
            reason: ChatGenerationFailureReason::MalformedModelOutput,
        })
        .await?;
    Ok(())
}

async fn send_progress_then_malformed_output_failure(
    request_id: RequestId,
    maximum_output_tokens: u16,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::GenerationProgress {
            request_id,
            generated_token_count: 3,
            maximum_output_tokens,
            elapsed_millis: 250,
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    tokio::time::sleep(Duration::from_millis(750)).await;
    send_malformed_output_failure(request_id, event_writer).await
}

async fn send_duplicate_generation_preparation(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    let generation_preparation_event = WorkerEvent::GenerationPreparationStarted {
        request_id,
        expert_residency: Some(ipc_protocol::WorkerExpertResidencySnapshot {
            dense_layers: None,
            total_layer_count: 40,
            resident_expert_count: 40,
            resident_expert_payload_bytes: 23_073_914_880,
            complete_expert_payload_byte_count: 23_073_914_880,
            expert_capacity: 256,
            layer_seating: ipc_protocol::ExpertLayerSeatingCounts::default(),
        }),
        mlx_memory_snapshot: None,
    };
    event_writer
        .send_event(&generation_preparation_event)
        .await?;
    event_writer
        .send_event(&generation_preparation_event)
        .await?;
    Ok(())
}

async fn send_out_of_order_output(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 1,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: "out of order".to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    Ok(())
}

async fn send_empty_output_batch(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: Vec::new(),
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    scripted_worker_chat::send_completion(
        request_id,
        1,
        1,
        ChatGenerationCompletionReason::EndOfSequence,
        event_writer,
    )
    .await?;
    Ok(())
}

async fn send_invalid_output_batch(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![
                ChatGenerationOutput::Text {
                    text: "must not escape malformed batch".to_owned(),
                },
                ChatGenerationOutput::ToolCall {
                    tool_call_index: 1,
                    function_name: "read".to_owned(),
                    arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
                },
            ],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    Ok(())
}

async fn send_over_budget_tool_completion(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::ToolCall {
                tool_call_index: 0,
                function_name: "read".to_owned(),
                arguments_json: r#"{"path":"AGENTS.md"}"#.to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    scripted_worker_chat::send_completion(
        request_id,
        1,
        17,
        ChatGenerationCompletionReason::ToolCalls,
        event_writer,
    )
    .await?;
    Ok(())
}

async fn send_backpressure_burst(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    for sequence_number in 0..64_u16 {
        event_writer
            .send_event(&WorkerEvent::Output {
                request_id,
                sequence_number,
                generated_token_count: 1,
                outputs: vec![ChatGenerationOutput::Text {
                    text: "fragment".to_owned(),
                }],
                mlx_memory_snapshot: None,
                expert_residency: None,
            })
            .await?;
    }
    scripted_worker_chat::send_completion(
        request_id,
        1,
        1,
        ChatGenerationCompletionReason::MaximumOutputTokens,
        event_writer,
    )
    .await?;
    Ok(())
}

async fn send_prefill_progress_reply(
    request_id: RequestId,
    event_writer: &mut ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
) -> Result<(), FixtureError> {
    event_writer
        .send_event(&WorkerEvent::PrefillProgress {
            request_id,
            prompt_processing_phase: WorkerPromptProcessingPhase::Target,
            processed_tokens: 2_048,
            total_tokens: 50_000,
            elapsed_millis: 1_500,
            forward_prefill_chunk_elapsed_millis: Some(1_400),
            completed_prefill_chunk_tokens: Some(2_048),
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::Prefill,
                active_memory_bytes: 11_000,
                mlx_buffer_cache_bytes: 12_000,
                peak_memory_bytes: 13_000,
                expert_payload_bytes: 4_000,
                model_core_payload_bytes: 3_000,
                context_state_payload_bytes: 2_000,
                memory_ceiling_utilization: None,
                memory_ledger: None,
            }),
            expert_residency: None,
            prompt_layer_progress: None,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::FirstDecodeCompleted {
            request_id,
            elapsed_millis: 321,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::Output {
            request_id,
            sequence_number: 0,
            generated_token_count: 1,
            outputs: vec![ChatGenerationOutput::Text {
                text: "done".to_owned(),
            }],
            mlx_memory_snapshot: None,
            expert_residency: None,
        })
        .await?;
    event_writer
        .send_event(&WorkerEvent::GenerationFinalized {
            request_id,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            expert_residency: None,
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::Finalized,
                active_memory_bytes: 24_000,
                mlx_buffer_cache_bytes: 0,
                peak_memory_bytes: 25_000,
                expert_payload_bytes: 19_000,
                model_core_payload_bytes: 3_000,
                context_state_payload_bytes: 0,
                memory_ceiling_utilization: None,
                memory_ledger: None,
            }),
        })
        .await?;
    scripted_worker_chat::send_completion(
        request_id,
        50_000,
        1,
        ChatGenerationCompletionReason::EndOfSequence,
        event_writer,
    )
    .await?;
    Ok(())
}
