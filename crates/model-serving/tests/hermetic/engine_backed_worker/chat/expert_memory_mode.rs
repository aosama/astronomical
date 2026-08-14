use super::*;

#[test]
fn should_replace_the_terminal_token_mode_with_the_post_finalization_mode() {
    let terminal_generated_token = GeneratedToken::TokenId {
        token_id: 1,
        is_reasoning_token: false,
        expert_memory_mode: Some(ExpertMemoryMode::Paged),
        mlx_memory_telemetry: None,
        first_decode_forward_elapsed_millis: None,
        generation_finalization: None,
    }
    .with_expert_memory_mode(Some(ExpertMemoryMode::Resident));

    assert_eq!(
        terminal_generated_token,
        GeneratedToken::TokenId {
            token_id: 1,
            is_reasoning_token: false,
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            mlx_memory_telemetry: None,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: None,
        }
    );
}

#[tokio::test]
async fn should_emit_only_changed_expert_memory_modes_without_losing_token_output() {
    let mut scripted_engine = ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
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
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(744, 12)))
        .await
        .expect("the worker should receive a chat request");
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Resident,
        }
    );
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged,
        }
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            persistent_prompt_cache_diagnostics: None,
            generated_token_count: 1,
            ..
        }
    ));

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_emit_the_recovered_resident_mode_before_completing_the_same_request() {
    let mut scripted_engine = ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
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
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(745, 2)))
        .await
        .expect("the worker should receive a chat request");
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Resident,
        }
    );
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged,
        }
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Resident,
        }
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 2,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            generated_token_count: 2,
            ..
        }
    ));

    close_worker_transport(supervisor_writer, worker_task).await;
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
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(746, 2)))
        .await
        .expect("the worker should receive the chat request");
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged,
        }
    );
    supervisor_writer
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(746),
        })
        .await
        .expect("the worker should receive cancellation");
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id: RequestId::new(746),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::Finalized,
                active_memory_bytes: 24_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 25_000,
                expert_payload_bytes: 19_000,
                model_core_payload_bytes: 3_000,
                context_state_payload_bytes: 0,
                speculative_prefill_draft_memory_bytes: 0,
            }),
            expert_residency: None,
        }
    );
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            request_id: RequestId::new(746),
            prompt_token_count: 1,
            generated_token_count: 0,
            reasoning_token_count: 0,
            cached_token_count: 0,
            persistent_prompt_cache_diagnostics: None,
            reason: ChatGenerationCompletionReason::Cancelled,
        }
    );

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_emit_finalized_residency_and_memory_before_normal_completion() {
    let final_mlx_memory_telemetry = MlxMemoryTelemetry::new(
        24_000,
        0,
        25_000,
        MlxActiveMemoryBreakdown {
            expert_payload_bytes: 19_000,
            model_core_payload_bytes: 3_000,
            context_state_payload_bytes: 0,
            speculative_prefill_draft_memory_bytes: 0,
        },
    );
    let mut scripted_engine = ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
        0,
        vec![GeneratedToken::TokenId {
            token_id: 1,
            is_reasoning_token: false,
            expert_memory_mode: Some(ExpertMemoryMode::Paged),
            mlx_memory_telemetry: None,
            first_decode_forward_elapsed_millis: None,
            generation_finalization: Some(GenerationFinalization::new(
                Some(ExpertMemoryMode::Resident),
                Some(final_mlx_memory_telemetry),
                None,
            )),
        }],
    );
    scripted_engine.initial_expert_memory_mode = Some(ExpertMemoryMode::Paged);
    let cancellation_count = scripted_engine.cancellation_count();
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(1),
        scripted_engine,
    );
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (supervisor_reader_transport, supervisor_writer_transport) = split(supervisor_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_reader_transport);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_writer_transport);
    let worker_task = tokio::spawn(async move {
        engine_worker
            .run(worker_reader_transport, worker_writer_transport)
            .await
    });

    assert_eq!(next_event(&mut supervisor_reader).await, ready_event());
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(747, 1)))
        .await
        .expect("the worker should receive the chat request");
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged,
        }
    );
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id: RequestId::new(747),
            expert_memory_mode: Some(ExpertMemoryMode::Resident),
            mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
                source: MlxMemorySnapshotSource::Finalized,
                active_memory_bytes: 24_000,
                allocator_cache_memory_bytes: 0,
                peak_memory_bytes: 25_000,
                expert_payload_bytes: 19_000,
                model_core_payload_bytes: 3_000,
                context_state_payload_bytes: 0,
                speculative_prefill_draft_memory_bytes: 0,
            }),
            expert_residency: None,
        }
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            generated_token_count: 1,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
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
    supervisor_writer
        .send_command(&WorkerCommand::Generate(chat_command(748, 2)))
        .await
        .expect("a terminally finalized engine should accept a follow-up request");
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::ExpertMemoryModeChanged {
            expert_memory_mode: ExpertMemoryMode::Paged
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::GenerationFinalized {
            request_id,
            ..
        } if request_id == RequestId::new(748)
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Output {
            request_id,
            generated_token_count: 1,
            ..
        } if request_id == RequestId::new(748)
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            request_id,
            generated_token_count: 1,
            ..
        } if request_id == RequestId::new(748)
    ));

    close_worker_transport(supervisor_writer, worker_task).await;
}
