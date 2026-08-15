use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{WorkerEvent, WorkerPromptProcessingPhase};

use crate::{
    ActiveRequestProgress, ChatGenerationStreamEvent, WorkerControlError, WorkerHealthSnapshot,
    chat_generation_executor::try_send_stream_event,
    prompt_processing_chunk_optimizer_status::record_prompt_processing_chunk_optimization_outcome,
    worker_health::{publish_active_request_progress, publish_latest_mlx_memory_snapshot},
    worker_loop_types::ActiveGeneration,
};

pub(super) fn handle_worker_prefill_progress(
    worker_prefill_progress_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_generation: &mut Option<ActiveGeneration>,
) -> Result<(), WorkerControlError> {
    let WorkerEvent::PrefillProgress {
        request_id,
        prompt_processing_phase,
        processed_tokens,
        total_tokens,
        elapsed_millis,
        forward_prefill_chunk_elapsed_millis,
        completed_prefill_chunk_tokens,
        prompt_processing_chunk_optimization_outcome,
        mlx_memory_snapshot,
        expert_residency,
        speculative_prefill_draft_memory_snapshot,
    } = worker_prefill_progress_event
    else {
        return Err(WorkerControlError::WorkerProtocolViolation {
            description: "prefill progress handler received another event",
        });
    };
    let Some(active_request) = active_generation.as_mut() else {
        return Ok(());
    };
    if request_id != active_request.request_id {
        return Ok(());
    }

    // Worker elapsed_millis is cumulative, so retain the latest amount.
    active_request.prefill_elapsed_millis = elapsed_millis;
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        active_request.maximum_mlx_peak_memory_bytes = Some(
            active_request
                .maximum_mlx_peak_memory_bytes
                .unwrap_or(0)
                .max(mlx_memory_snapshot.peak_memory_bytes),
        );
        active_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
    }
    let latest_mlx_memory_snapshot =
        speculative_prefill_draft_memory_snapshot.or(mlx_memory_snapshot);
    if let Some(latest_mlx_memory_snapshot) = latest_mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, latest_mlx_memory_snapshot);
    }
    if let Some(expert_residency) = expert_residency {
        crate::worker_health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    try_send_stream_event(
        &active_request.stream_event_sender,
        ChatGenerationStreamEvent::PrefillProgress {
            processed_tokens,
            total_tokens,
            elapsed_millis,
            forward_prefill_chunk_elapsed_millis,
            completed_prefill_chunk_tokens,
            mlx_active_memory_bytes: mlx_memory_snapshot
                .map(|snapshot| snapshot.active_memory_bytes),
            mlx_allocator_cache_memory_bytes: mlx_memory_snapshot
                .map(|snapshot| snapshot.allocator_cache_memory_bytes),
            mlx_peak_memory_bytes: mlx_memory_snapshot.map(|snapshot| snapshot.peak_memory_bytes),
        },
    )?;
    let published_target_processed_tokens = match prompt_processing_phase {
        WorkerPromptProcessingPhase::Drafter => 0,
        WorkerPromptProcessingPhase::Target => processed_tokens,
    };
    publish_active_request_progress(
        health_snapshot,
        ActiveRequestProgress::Prefill {
            prompt_processing_phase,
            processed_tokens: published_target_processed_tokens,
            total_tokens,
            request_started_at: active_request.request_started_at,
            elapsed_millis,
            completed_prefill_chunk_tokens,
        },
    );
    if let Some(prompt_processing_chunk_optimization_outcome) =
        prompt_processing_chunk_optimization_outcome
    {
        record_prompt_processing_chunk_optimization_outcome(
            health_snapshot,
            prompt_processing_chunk_optimization_outcome,
        );
    }
    Ok(())
}
