use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, ProtocolReader, ProtocolWriter, RequestId, WorkerEvent,
    WorkerExpertResidencySnapshot, WorkerMlxMemorySnapshot, WorkerPromptProcessingPhase,
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
            resident_expert_count: 40,
            resident_expert_payload_bytes: 3_375_366_144,
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
async fn should_round_trip_prefill_progress_before_the_first_completed_forward() {
    let worker_event = WorkerEvent::PrefillProgress {
        request_id: RequestId::new(82),
        prompt_processing_phase: WorkerPromptProcessingPhase::Drafter,
        processed_tokens: 0,
        total_tokens: 4_096,
        elapsed_millis: 0,
        forward_prefill_chunk_elapsed_millis: None,
        completed_prefill_chunk_tokens: None,
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
