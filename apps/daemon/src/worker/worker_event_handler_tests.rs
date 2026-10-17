//! Unit contracts for process-scoped worker-event handling that touches
//! request-scoped state.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{
    MlxMemorySnapshotSource, RequestId, WorkerEvent, WorkerMlxMemorySnapshot,
};
use tokio::sync::{Semaphore, mpsc};

use crate::worker::worker_event_handler::handle_worker_event;
use crate::worker::worker_loop_types::{ActiveGeneration, ActiveWorkerRequest};
use crate::{
    CompletionAttributionLog, GenerationPerformanceLog, WorkerHealthSnapshot, WorkerHealthStatus,
};

fn standalone_sample(peak_memory_bytes: u64) -> WorkerMlxMemorySnapshot {
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

fn active_chat_request(
    stream_event_sender: mpsc::Sender<crate::ChatGenerationStreamEvent>,
    maximum_output_tokens: u16,
) -> ActiveWorkerRequest {
    let active_generation_permit = Arc::new(Semaphore::new(1))
        .try_acquire_owned()
        .expect("the test semaphore permit should be acquirable");
    ActiveWorkerRequest::Chat(ActiveGeneration {
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
    })
}

#[test]
fn should_fold_a_standalone_memory_sample_peak_into_the_active_chat_request() {
    let temporary_log_directory = tempfile::tempdir().expect("temporary log directory");
    let mut performance_log =
        GenerationPerformanceLog::open(temporary_log_directory.path()).expect("performance log");
    let mut completion_log = CompletionAttributionLog::disabled();
    let health_snapshot = Arc::new(RwLock::new(WorkerHealthSnapshot::unavailable(
        WorkerHealthStatus::Loading,
    )));
    let (stream_event_sender, _stream_event_receiver) = mpsc::channel(16);
    let mut active_request = Some(active_chat_request(stream_event_sender, 200));
    let sample_event = WorkerEvent::MlxMemorySample {
        mlx_memory_snapshot: Some(standalone_sample(37_398_417_028)),
        expert_residency: None,
    };

    handle_worker_event(
        sample_event,
        &health_snapshot,
        &mut true,
        &mut None,
        &mut active_request,
        &mut performance_log,
        &mut completion_log,
    )
    .expect("the standalone sample event should be accepted");

    let folded_peak = match active_request {
        Some(ActiveWorkerRequest::Chat(chat_request)) => chat_request.maximum_mlx_peak_memory_bytes,
        _ => panic!("the chat request should remain active"),
    };
    assert_eq!(folded_peak, Some(37_398_417_028));
}
