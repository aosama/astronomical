use super::*;

#[tokio::test]
async fn should_exclude_the_complete_restored_prompt_prefix_from_progress_without_inflating_cached_usage()
 {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(15),
        ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
            3,
            vec![
                GeneratedToken::PrefillProgress {
                    persistent_prompt_cache_diagnostics: None,
                    processed_token_count: 2,
                    elapsed_millis: 400,
                    forward_prefill_chunck_elapsed_millis: 350,
                    completed_prefill_chunck_tokens: 2_048,
                    prefill_optimizer_insight: None,
                    mlx_memory_telemetry: Some(MlxMemoryTelemetry::new(
                        11_000,
                        12_000,
                        13_000,
                        MlxActiveMemoryBreakdown {
                            expert_payload_bytes: 1_000,
                            model_core_payload_bytes: 4_000,
                            context_state_payload_bytes: 2_000,
                            speculative_prefill_draft_memory_bytes: 0,
                        },
                    )),
                    speculative_prefill_draft_memory_telemetry: Some(MlxMemoryTelemetry::new(
                        20_000,
                        1_000,
                        22_000,
                        MlxActiveMemoryBreakdown {
                            expert_payload_bytes: 2_000,
                            model_core_payload_bytes: 3_000,
                            context_state_payload_bytes: 1_000,
                            speculative_prefill_draft_memory_bytes: 14_000,
                        },
                    )),
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse {
                        target_eligible_token_count: 13,
                        target_restored_token_count: 10,
                        drafter_eligible_token_count: 15,
                        drafter_restored_token_count: 10,
                    },
                },
                GeneratedToken::PrefillProgress {
                    persistent_prompt_cache_diagnostics: None,
                    processed_token_count: 3,
                    elapsed_millis: 600,
                    forward_prefill_chunck_elapsed_millis: 525,
                    completed_prefill_chunck_tokens: 2_048,
                    prefill_optimizer_insight: None,
                    mlx_memory_telemetry: Some(MlxMemoryTelemetry::new(
                        14_000,
                        15_000,
                        16_000,
                        MlxActiveMemoryBreakdown::default(),
                    )),
                    speculative_prefill_draft_memory_telemetry: None,
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse {
                        target_eligible_token_count: 13,
                        target_restored_token_count: 10,
                        drafter_eligible_token_count: 15,
                        drafter_restored_token_count: 10,
                    },
                },
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    generation_finalization: None,
                },
            ],
        )
        .with_restored_prompt_prefix_token_count(10),
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
        .send_command(&WorkerCommand::Generate(chat_command(742, 12)))
        .await
        .expect("the worker should receive a chat request");

    let initial_prefill_progress_event = next_event(&mut supervisor_reader).await;
    match initial_prefill_progress_event {
        WorkerEvent::PrefillProgress {
            request_id,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            completed_prefill_chunck_tokens,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 0);
            assert_eq!(total_tokens, 5);
            assert_eq!(elapsed_millis, 0);
            assert_eq!(completed_prefill_chunck_tokens, None);
        }
        other_event => panic!("expected initial uncached prefill progress, got {other_event:?}"),
    }

    let first_completed_prefill_progress_event = next_event(&mut supervisor_reader).await;
    match first_completed_prefill_progress_event {
        WorkerEvent::PrefillProgress {
            request_id,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            forward_prefill_chunck_elapsed_millis,
            completed_prefill_chunck_tokens,
            prefill_optimizer_insight,
            mlx_memory_snapshot,
            speculative_prefill_draft_memory_snapshot,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 2);
            assert_eq!(total_tokens, 3);
            assert_eq!(elapsed_millis, 400);
            assert_eq!(forward_prefill_chunck_elapsed_millis, Some(350));
            assert_eq!(completed_prefill_chunck_tokens, Some(2_048));
            assert_eq!(prefill_optimizer_insight, None);
            assert_eq!(
                mlx_memory_snapshot,
                Some(WorkerMlxMemorySnapshot {
                    source: MlxMemorySnapshotSource::Prefill,
                    active_memory_bytes: 11_000,
                    allocator_cache_memory_bytes: 12_000,
                    peak_memory_bytes: 13_000,
                    expert_payload_bytes: 1_000,
                    model_core_payload_bytes: 4_000,
                    context_state_payload_bytes: 2_000,
                    speculative_prefill_draft_memory_bytes: 0,
                })
            );
            assert_eq!(
                speculative_prefill_draft_memory_snapshot,
                Some(WorkerMlxMemorySnapshot {
                    source: MlxMemorySnapshotSource::SpeculativePrefillDraftScoring,
                    active_memory_bytes: 20_000,
                    allocator_cache_memory_bytes: 1_000,
                    peak_memory_bytes: 22_000,
                    expert_payload_bytes: 2_000,
                    model_core_payload_bytes: 3_000,
                    context_state_payload_bytes: 1_000,
                    speculative_prefill_draft_memory_bytes: 14_000,
                })
            );
        }
        other_event => panic!("expected uncached prefill progress, got {other_event:?}"),
    }
    let second_completed_prefill_progress_event = next_event(&mut supervisor_reader).await;
    match second_completed_prefill_progress_event {
        WorkerEvent::PrefillProgress {
            request_id,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            completed_prefill_chunck_tokens,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 3);
            assert_eq!(total_tokens, 3);
            assert_eq!(elapsed_millis, 1_000);
            assert_eq!(completed_prefill_chunck_tokens, Some(2_048));
        }
        other_event => panic!("expected uncached prefill progress, got {other_event:?}"),
    }
    let _first_output_event = next_event(&mut supervisor_reader).await;
    let _second_output_event = next_event(&mut supervisor_reader).await;
    assert_eq!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::PromptWorkReuse {
            request_id: RequestId::new(742),
            prompt_work_reuse: WorkerPromptWorkReuse {
                target_eligible_token_count: 13,
                target_restored_token_count: 10,
                drafter_eligible_token_count: 15,
                drafter_restored_token_count: 10,
            },
        }
    );
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::Completed {
            prompt_token_count: 15,
            cached_token_count: 3,
            ..
        }
    ));

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_report_completed_prefill_chunck_tokens_after_measurement() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(15),
        ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
            10,
            vec![
                GeneratedToken::PrefillProgress {
                    persistent_prompt_cache_diagnostics: None,
                    processed_token_count: 2,
                    elapsed_millis: 400,
                    forward_prefill_chunck_elapsed_millis: 350,
                    completed_prefill_chunck_tokens: 512,
                    prefill_optimizer_insight: None,
                    mlx_memory_telemetry: None,
                    speculative_prefill_draft_memory_telemetry: None,
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse::default(),
                },
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    generation_finalization: None,
                },
            ],
        ),
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
        .send_command(&WorkerCommand::Generate(chat_command(743, 12)))
        .await
        .expect("the worker should receive a chat request");

    let initial_prefill_progress_event = next_event(&mut supervisor_reader).await;
    assert!(
        matches!(
            initial_prefill_progress_event,
            WorkerEvent::PrefillProgress {
                completed_prefill_chunck_tokens: None,
                ..
            }
        ),
        "the initial event should still report the configured maximum before any measured chunk: {initial_prefill_progress_event:?}"
    );

    let measured_prefill_progress_event = next_event(&mut supervisor_reader).await;
    match measured_prefill_progress_event {
        WorkerEvent::PrefillProgress {
            request_id,
            processed_tokens,
            total_tokens,
            elapsed_millis,
            completed_prefill_chunck_tokens,
            mlx_memory_snapshot,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(743));
            assert_eq!(processed_tokens, 2);
            assert_eq!(total_tokens, 5);
            assert_eq!(elapsed_millis, 400);
            assert_eq!(completed_prefill_chunck_tokens, Some(512));
            assert_eq!(mlx_memory_snapshot, None);
        }
        other_event => panic!("expected measured active prefill progress, got {other_event:?}"),
    }

    close_worker_transport(supervisor_writer, worker_task).await;
}

#[tokio::test]
async fn should_report_only_the_confirmed_active_prompt_processing_phase() {
    let engine_worker = EngineBackedWorker::new(
        ScriptedChatProcessor::with_prompt_token_count(16),
        ScriptedChatEngine::with_cached_token_count_and_generated_tokens(
            0,
            vec![
                GeneratedToken::PromptProcessingPhaseStarted {
                    prompt_processing_phase: WorkerPromptProcessingPhase::Drafter,
                    total_token_count: 16,
                },
                GeneratedToken::PrefillProgress {
                    persistent_prompt_cache_diagnostics: None,
                    processed_token_count: 8,
                    elapsed_millis: 400,
                    forward_prefill_chunck_elapsed_millis: 350,
                    completed_prefill_chunck_tokens: 8,
                    prefill_optimizer_insight: None,
                    mlx_memory_telemetry: None,
                    speculative_prefill_draft_memory_telemetry: None,
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse::default(),
                },
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    generation_finalization: None,
                },
            ],
        ),
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
        .expect("the worker should receive a phase-aware chat request");

    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::PrefillProgress {
            prompt_processing_phase: WorkerPromptProcessingPhase::Target,
            processed_tokens: 0,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::PrefillProgress {
            prompt_processing_phase: WorkerPromptProcessingPhase::Drafter,
            processed_tokens: 0,
            total_tokens: 16,
            ..
        }
    ));
    assert!(matches!(
        next_event(&mut supervisor_reader).await,
        WorkerEvent::PrefillProgress {
            prompt_processing_phase: WorkerPromptProcessingPhase::Target,
            processed_tokens: 8,
            ..
        }
    ));

    supervisor_writer
        .send_command(&WorkerCommand::Cancel {
            request_id: RequestId::new(744),
        })
        .await
        .expect("the phase-aware request should be cancellable");
    close_worker_transport(supervisor_writer, worker_task).await;
}
