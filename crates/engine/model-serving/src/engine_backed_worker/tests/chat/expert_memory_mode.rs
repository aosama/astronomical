use super::chat_request_exchange::{
    expect_expert_memory_mode_changed, plain_completed_event, send_cancellation, send_chat_request,
};
use super::ipc_worker_transport::open_ipc_worker_transport;
use super::scripted_chat_test_doubles::{ScriptedChatEngine, ScriptedChatProcessor};
use super::support::{close_worker_transport, next_event, ready_event};
use crate::{
    EngineBackedWorker, GeneratedToken, GenerationFinalization, MlxActiveMemoryBreakdown,
    MlxMemoryTelemetry,
};
use ipc_protocol::{
    ChatGenerationCompletionReason, ExpertMemoryMode, MlxMemorySnapshotSource, ProtocolReader,
    ProtocolWriter, RequestId, WorkerEvent, WorkerMlxMemorySnapshot,
};
#[tokio::test]
async fn should_emit_only_changed_expert_memory_modes_without_losing_token_output() {
    let mut scripted_engine =
        ScriptedChatEngine::with_astro_ssd_prompt_cache_restored_token_count_and_generated_tokens(
            0,
            vec![GeneratedToken::TokenId {
                token_id: 1,
                is_reasoning_token: false,
                expert_memory_mode: Some(ExpertMemoryMode::Paged),
                mlx_memory_telemetry: None,
                first_decode_forward_elapsed_millis: None,
                generation_finalization: None,
            }],
        );
    scripted_engine.initial_expert_memory_mode = Some(ExpertMemoryMode::Resident);
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(1),
        scripted_engine,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Resident).await;
    send_chat_request(&mut daemon_writer, 744, 12).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Resident).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            astro_ssd_prompt_cache_diagnostics: None,
            generated_token_count: 1,
            ..
        }
    ));

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_emit_the_recovered_resident_mode_before_completing_the_same_request() {
    let mut scripted_engine =
        ScriptedChatEngine::with_astro_ssd_prompt_cache_restored_token_count_and_generated_tokens(
            0,
            vec![
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: Some(ExpertMemoryMode::Paged),
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: Some(ExpertMemoryMode::Resident),
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
            ],
        );
    scripted_engine.initial_expert_memory_mode = Some(ExpertMemoryMode::Resident);
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(1),
        scripted_engine,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Resident).await;
    send_chat_request(&mut daemon_writer, 745, 2).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Resident).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Resident).await;
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            generated_token_count: 2,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            generated_token_count: 2,
            ..
        }
    ));

    close_worker_transport(daemon_writer, worker_task).await;
}

#[tokio::test]
async fn should_emit_finalized_residency_and_memory_before_cancellation_completion() {
    let mut scripted_engine = ScriptedChatEngine::with_cancelled_generation_finalization(
        ExpertMemoryMode::Resident,
        24_000,
        0,
        25_000,
        19_000,
        3_000,
        0,
    );
    scripted_engine.initial_expert_memory_mode = Some(ExpertMemoryMode::Paged);
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(1),
        scripted_engine,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    send_chat_request(&mut daemon_writer, 746, 2).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    send_cancellation(&mut daemon_writer, 746).await;
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id: RequestId::new(746),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
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
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut daemon_reader).await,
        plain_completed_event(746, 1, 0, ChatGenerationCompletionReason::Cancelled)
    );

    close_worker_transport(daemon_writer, worker_task).await;
}

/// One scripted decode token that finalizes the generation to Resident mode
/// with a fully populated memory telemetry snapshot, so the wire events can
/// prove finalization lands before the normal completion.
fn finalized_resident_generation_token() -> GeneratedToken {
    let final_mlx_memory_telemetry = MlxMemoryTelemetry::new(
        24_000,
        0,
        25_000,
        MlxActiveMemoryBreakdown {
            expert_payload_bytes: 19_000,
            model_core_payload_bytes: 3_000,
            context_state_payload_bytes: 0,
        },
    );
    GeneratedToken::TokenId {
        token_id: 1,
        is_reasoning_token: false,
        expert_memory_mode: Some(ExpertMemoryMode::Paged),
        mlx_memory_telemetry: None,
        first_decode_forward_elapsed_millis: None,
        generation_finalization: Some(Box::new(GenerationFinalization::new(
            Some(ExpertMemoryMode::Resident),
            Some(final_mlx_memory_telemetry),
            None,
        ))),
    }
}

/// Proves a terminally finalized engine accepts a follow-up request and
/// repeats the mode-changed, finalized, output, completed sequence for it.
async fn assert_follow_up_request_after_terminal_finalization<
    ReadTransport: tokio::io::AsyncRead + Unpin,
    WriteTransport: tokio::io::AsyncWrite + Unpin,
>(
    daemon_reader: &mut ProtocolReader<ReadTransport>,
    daemon_writer: &mut ProtocolWriter<WriteTransport>,
) {
    send_chat_request(daemon_writer, 748, 2).await;
    assert!(matches!(
        next_event(daemon_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged
        }
    ));
    assert!(matches!(
        next_event(daemon_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id,
            ..
        } if request_id == RequestId::new(748)
    ));
    assert!(matches!(
        next_event(daemon_reader).await,
        WorkerEvent::Output {
            request_id,
            generated_token_count: 1,
            ..
        } if request_id == RequestId::new(748)
    ));
    assert!(matches!(
        next_event(daemon_reader).await,
        WorkerEvent::Completed {
            request_id,
            generated_token_count: 1,
            ..
        } if request_id == RequestId::new(748)
    ));
}

#[tokio::test]
async fn should_emit_finalized_residency_and_memory_before_normal_completion() {
    let mut scripted_engine =
        ScriptedChatEngine::with_astro_ssd_prompt_cache_restored_token_count_and_generated_tokens(
            0,
            vec![finalized_resident_generation_token()],
        );
    scripted_engine.initial_expert_memory_mode = Some(ExpertMemoryMode::Paged);
    let cancellation_count = scripted_engine.cancellation_count();
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(1),
        scripted_engine,
    );
    let (mut daemon_reader, mut daemon_writer, worker_reader_transport, worker_writer_transport) =
        open_ipc_worker_transport();
    let worker_task =
        tokio::spawn(engine_worker.run(worker_reader_transport, worker_writer_transport));

    assert_eq!(next_event(&mut daemon_reader).await, ready_event());
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    send_chat_request(&mut daemon_writer, 747, 1).await;
    expect_expert_memory_mode_changed(&mut daemon_reader, ExpertMemoryMode::Paged).await;
    assert_eq!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id: RequestId::new(747),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
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
            expert_residency: None,
        }
    );
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut daemon_reader).await,
        WorkerEvent::Completed {
            generated_token_count: 1,
            ..
        }
    ));
    assert_eq!(
        cancellation_count.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "an engine that already emitted terminal finalization must not receive a redundant cancellation"
    );
    assert_follow_up_request_after_terminal_finalization(&mut daemon_reader, &mut daemon_writer)
        .await;

    close_worker_transport(daemon_writer, worker_task).await;
}
