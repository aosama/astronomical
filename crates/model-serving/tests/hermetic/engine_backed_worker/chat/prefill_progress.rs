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
                    forward_prefill_chunk_elapsed_millis: 350,
                    completed_prefill_chunk_tokens: 2_048,
                    expert_residency_telemetry: Some(ExpertResidencyTelemetry {
                        total_layer_count: 40,
                        resident_expert_count: 40,
                        resident_expert_payload_bytes: 3_375_366_144,
                    }),
                    mlx_memory_telemetry: Some(MlxMemoryTelemetry::new(
                        11_000,
                        12_000,
                        13_000,
                        MlxActiveMemoryBreakdown {
                            expert_payload_bytes: 1_000,
                            model_core_payload_bytes: 4_000,
                            context_state_payload_bytes: 2_000,
                        },
                    )),
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse {
                        target_eligible_token_count: 13,
                        target_restored_token_count: 10,
                    },
                },
                GeneratedToken::PrefillProgress {
                    persistent_prompt_cache_diagnostics: None,
                    processed_token_count: 3,
                    elapsed_millis: 600,
                    forward_prefill_chunk_elapsed_millis: 525,
                    completed_prefill_chunk_tokens: 2_048,
                    expert_residency_telemetry: None,
                    mlx_memory_telemetry: Some(MlxMemoryTelemetry::new(
                        14_000,
                        15_000,
                        16_000,
                        MlxActiveMemoryBreakdown::default(),
                    )),
                    expert_memory_mode: None,
                    prompt_work_reuse: WorkerPromptWorkReuse {
                        target_eligible_token_count: 13,
                        target_restored_token_count: 10,
                    },
                },
                GeneratedToken::TokenId {
                    token_id: 1,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
                    generation_finalization: None,
                },
                GeneratedToken::TokenId {
                    token_id: 2,
                    is_reasoning_token: false,
                    expert_memory_mode: None,
                    mlx_memory_telemetry: None,
                    first_decode_forward_elapsed_millis: None,
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
            completed_prefill_chunk_tokens,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 0);
            assert_eq!(total_tokens, 5);
            assert_eq!(elapsed_millis, 0);
            assert_eq!(completed_prefill_chunk_tokens, None);
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
            forward_prefill_chunk_elapsed_millis,
            completed_prefill_chunk_tokens,
            mlx_memory_snapshot,
            expert_residency,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 2);
            assert_eq!(total_tokens, 3);
            assert_eq!(elapsed_millis, 400);
            assert_eq!(forward_prefill_chunk_elapsed_millis, Some(350));
            assert_eq!(completed_prefill_chunk_tokens, Some(2_048));
            assert_eq!(
                expert_residency,
                Some(WorkerExpertResidencySnapshot {
                    total_layer_count: 40,
                    resident_expert_count: 40,
                    resident_expert_payload_bytes: 3_375_366_144,
                })
            );
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
                    memory_ceiling_utilization: None,
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
            completed_prefill_chunk_tokens,
            ..
        } => {
            assert_eq!(request_id, RequestId::new(742));
            assert_eq!(processed_tokens, 3);
            assert_eq!(total_tokens, 3);
            assert_eq!(elapsed_millis, 1_000);
            assert_eq!(completed_prefill_chunk_tokens, Some(2_048));
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
