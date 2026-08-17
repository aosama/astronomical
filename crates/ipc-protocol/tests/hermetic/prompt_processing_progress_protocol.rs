use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, ProtocolReader, ProtocolWriter, RequestId, WorkerEvent,
    WorkerExpertResidencySnapshot, WorkerMlxMemorySnapshot,
    WorkerPromptProcessingChunkCandidateMeasurementSummary,
    WorkerPromptProcessingChunkMeasurementSource, WorkerPromptProcessingChunkOptimizationContext,
    WorkerPromptProcessingChunkOptimizationOutcome, WorkerPromptProcessingChunkSelectionReason,
    WorkerPromptProcessingPhase,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_prefill_progress_event() {
    let worker_event = WorkerEvent::PrefillProgress {
        request_id: RequestId::new(81),
        prompt_processing_phase: WorkerPromptProcessingPhase::Target,
        processed_tokens: 1_024,
        total_tokens: 4_096,
        elapsed_millis: 1_200,
        forward_prefill_chunk_elapsed_millis: Some(1_100),
        completed_prefill_chunk_tokens: Some(2_048),
        prompt_processing_chunk_optimization_outcome: Some(
            WorkerPromptProcessingChunkOptimizationOutcome {
                selected_candidate_chunk_size_tokens: 4_096,
                processed_prompt_token_count: 2_048,
                forward_elapsed_millis: 1_200,
                was_reduced_by_memory_capacity: true,
                was_accepted_for_learning: false,
                selection_reason: WorkerPromptProcessingChunkSelectionReason::MinimizeProjectedRemainingPromptLatency,
                measurement_context: WorkerPromptProcessingChunkOptimizationContext {
                    chunk_start_token_position: 8_192,
                    position_range_start_token_position: 0,
                    position_range_end_token_position_exclusive: 32_768,
                    has_restored_prefix: false,
                    is_first_chunk_after_restore: false,
                    has_visual_embeddings: false,
                    is_mtp_active: false,
                    are_sparse_experts_paged: true,
                    is_prompt_cache_capture_eligible: true,
                    has_prior_capacity_reduction: false,
                },
                all_candidates_have_measurements: true,
                is_execution_profile_converged: true,
                candidate_measurement_summaries: vec![
                    WorkerPromptProcessingChunkCandidateMeasurementSummary {
                        candidate_chunk_size_tokens: 4_096,
                        measurement_source: WorkerPromptProcessingChunkMeasurementSource::ExecutionProfile,
                        measurement_count: 3,
                        average_processed_prompt_token_count: 3_413,
                        average_forward_elapsed_millis: 900,
                    },
                ],
            },
        ),
        mlx_memory_snapshot: Some(WorkerMlxMemorySnapshot {
            source: MlxMemorySnapshotSource::Prefill,
            active_memory_bytes: 11_000,
            allocator_cache_memory_bytes: 12_000,
            peak_memory_bytes: 13_000,
            expert_payload_bytes: 4_000,
            model_core_payload_bytes: 3_000,
            context_state_payload_bytes: 2_000,
            speculative_prefill_draft_memory_bytes: 0,
        }),
        expert_residency: Some(WorkerExpertResidencySnapshot {
            total_layer_count: 40,
            complete_layer_count: 6,
            complete_layer_payload_bytes: 2_852_126_720,
            partial_layer_count: 34,
            partial_layer_payload_bytes: 523_239_424,
        }),
        speculative_prefill_draft_memory_snapshot: Some(WorkerMlxMemorySnapshot {
            source: MlxMemorySnapshotSource::SpeculativePrefillDraftScoring,
            active_memory_bytes: 20_000,
            allocator_cache_memory_bytes: 1_000,
            peak_memory_bytes: 22_000,
            expert_payload_bytes: 2_000,
            model_core_payload_bytes: 3_000,
            context_state_payload_bytes: 1_000,
            speculative_prefill_draft_memory_bytes: 14_000,
        }),
    };

    assert_round_tripped_worker_event(worker_event).await;
}

#[tokio::test]
async fn should_round_trip_prefill_progress_before_a_chunk_measurement() {
    let worker_event = WorkerEvent::PrefillProgress {
        request_id: RequestId::new(82),
        prompt_processing_phase: WorkerPromptProcessingPhase::Drafter,
        processed_tokens: 0,
        total_tokens: 4_096,
        elapsed_millis: 0,
        forward_prefill_chunk_elapsed_millis: None,
        completed_prefill_chunk_tokens: None,
        prompt_processing_chunk_optimization_outcome: None,
        mlx_memory_snapshot: None,
        expert_residency: None,
        speculative_prefill_draft_memory_snapshot: None,
    };

    assert_round_tripped_worker_event(worker_event).await;
}

async fn assert_round_tripped_worker_event(worker_event: WorkerEvent) {
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("a prompt-processing progress event should be written");
    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the prompt-processing progress event should decode"),
        Some(worker_event)
    );
}
