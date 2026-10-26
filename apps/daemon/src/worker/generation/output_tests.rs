//! Unit contracts for folding decode-phase MLX snapshot peaks into the
//! request's durable peak record.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    ChatGenerationOutput, MlxMemorySnapshotSource, RequestId, WorkerEvent, WorkerMlxMemorySnapshot,
};
use tokio::sync::{Semaphore, mpsc};

use crate::WorkerHealthSnapshot;
use crate::worker::generation::output::{handle_worker_generation_progress, handle_worker_output};
use crate::worker::loop_types::ActiveGeneration;

fn decode_snapshot(peak_memory_bytes: u64) -> WorkerMlxMemorySnapshot {
    WorkerMlxMemorySnapshot {
        source: MlxMemorySnapshotSource::DecodeSubmitted,
        active_memory_bytes: peak_memory_bytes / 2,
        allocator_cache_memory_bytes: 0,
        peak_memory_bytes,
        expert_payload_bytes: 0,
        model_core_payload_bytes: 0,
        context_state_payload_bytes: 0,
        memory_ceiling_utilization: None,
    }
}

fn active_generation_with_token_budget(
    stream_event_sender: mpsc::Sender<crate::ChatGenerationStreamEvent>,
    maximum_output_tokens: u16,
) -> ActiveGeneration {
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    ActiveGeneration {
        _active_generation_permit: active_generation_permit,
        generated_token_count: 0,
        generation_started_at: None,
        generation_preparation_started_at: None,
        generation_preparation_elapsed_millis: None,
        first_decode_forward_elapsed_millis: None,
        time_to_first_output_millis: None,
        final_resident_expert_count: None,
        final_resident_expert_payload_bytes: None,
        latest_generation_progress_token_count: 0,
        max_output_tokens: maximum_output_tokens,
        next_sequence_number: 0,
        next_tool_call_index: 0,
        completed_tool_calls: Vec::new(),
        request_started_at: tokio::time::Instant::now(),
        prefill_elapsed_millis: 0,
        maximum_mlx_peak_memory_bytes: None,
        last_mlx_active_memory_bytes: None,
        request_id: RequestId::new(1),
        stream_event_sender,
    }
}

fn health_snapshot() -> Arc<RwLock<WorkerHealthSnapshot>> {
    Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        crate::WorkerHealthStatus::Loading,
    )))
}

#[test]
fn should_fold_the_decode_output_snapshot_peak_into_the_request_maximum() {
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    let mut active_generation = active_generation_with_token_budget(stream_event_sender, 200);
    let output_event = WorkerEvent::Output {
        request_id: RequestId::new(1),
        sequence_number: 0,
        generated_token_count: 1,
        outputs: vec![ChatGenerationOutput::Text {
            text: "hello".to_owned(),
        }],
        mlx_memory_snapshot: Some(decode_snapshot(37_398_417_028)),
        expert_residency: None,
    };

    handle_worker_output(output_event, &health_snapshot(), &mut active_generation)
        .expect("the correlated output event should be accepted");

    assert_eq!(
        active_generation.maximum_mlx_peak_memory_bytes,
        Some(37_398_417_028)
    );
    assert_eq!(
        active_generation.last_mlx_active_memory_bytes,
        Some(37_398_417_028 / 2)
    );
}

#[test]
fn should_retain_the_highest_decode_peak_across_lower_later_snapshots() {
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    let mut active_generation = active_generation_with_token_budget(stream_event_sender, 200);
    active_generation.maximum_mlx_peak_memory_bytes = Some(37_398_417_028);
    let output_event = WorkerEvent::Output {
        request_id: RequestId::new(1),
        sequence_number: 0,
        generated_token_count: 1,
        outputs: vec![ChatGenerationOutput::Text {
            text: "hello".to_owned(),
        }],
        // The MLX peak counter resets at allocator-cache cleanups, so a later
        // decode sample can observe a lower window peak than an earlier one.
        mlx_memory_snapshot: Some(decode_snapshot(35_293_875_951)),
        expert_residency: None,
    };

    handle_worker_output(output_event, &health_snapshot(), &mut active_generation)
        .expect("the correlated output event should be accepted");

    assert_eq!(
        active_generation.maximum_mlx_peak_memory_bytes,
        Some(37_398_417_028)
    );
}

#[test]
fn should_fold_the_generation_progress_snapshot_peak_into_the_request_maximum() {
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    let mut active_generation = active_generation_with_token_budget(stream_event_sender, 200);
    active_generation.maximum_mlx_peak_memory_bytes = Some(30_000_000_000);
    let progress_event = WorkerEvent::GenerationProgress {
        request_id: RequestId::new(1),
        generated_token_count: 5,
        maximum_output_tokens: 200,
        elapsed_millis: 250,
        mlx_memory_snapshot: Some(decode_snapshot(36_954_430_548)),
        expert_residency: None,
    };

    handle_worker_generation_progress(progress_event, &health_snapshot(), &mut active_generation)
        .expect("the correlated progress event should be accepted");

    assert_eq!(
        active_generation.maximum_mlx_peak_memory_bytes,
        Some(36_954_430_548)
    );
}
