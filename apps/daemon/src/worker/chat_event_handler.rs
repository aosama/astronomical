//! Applies the worker events of one chat generation's lifecycle to the
//! active chat request and the published health: its progress, its
//! finalization, its completion and its failure.
//!
//! The dispatch in `event_handler` routes only these events here.

use std::sync::{Arc, RwLock};

use ipc_protocol::WorkerEvent;

use crate::{
    ChatGenerationStreamEvent, CompletionAttributionLog, GenerationPerformanceLog, WorkerActivity,
    WorkerControlError, WorkerHealthSnapshot,
    serving::chat_generation_executor::try_send_stream_event,
    worker::event_handler::protocol_violation,
    worker::generation::completion_event::handle_worker_completion_event,
    worker::generation::output::{
        handle_worker_first_decode_completed, handle_worker_generation_progress,
        handle_worker_output, handle_worker_reply_report,
    },
    worker::generation::prefill_progress::handle_worker_prefill_progress,
    worker::generation::preparation::handle_generation_preparation_started,
    worker::health::{
        clear_active_request_progress, clear_latest_mlx_memory_snapshot, publish_activity,
        publish_expert_memory_mode, publish_latest_mlx_memory_snapshot,
    },
    worker::loop_types::{ActiveGeneration, ActiveWorkerRequest},
};

/// Applies the events of one chat generation's lifecycle: its finalization,
/// its progress, its completion and its failure. The main dispatch match
/// routes only these events here.
pub(super) fn dispatch_chat_generation_event(
    chat_generation_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    match chat_generation_event {
        generation_finalized_event @ WorkerEvent::GenerationFinalized { .. } => {
            apply_generation_finalized(generation_finalized_event, health_snapshot, active_request)
        }
        worker_completed_event @ WorkerEvent::Completed { .. } => apply_worker_completed(
            worker_completed_event,
            health_snapshot,
            active_request,
            performance_log,
            completion_log,
        ),
        generation_failed_event @ WorkerEvent::Failed { .. } => {
            apply_generation_failed(generation_failed_event, health_snapshot, active_request)
        }
        // What remains is output, prefill, preparation, generation progress,
        // first decode, prompt work reuse and the drafting report.
        worker_progress_event => {
            dispatch_chat_progress_event(worker_progress_event, health_snapshot, active_request)
        }
    }
}

/// Applies a generation-finalized event to the matching active chat request:
/// final peaks, last active memory, expert mode, and final residency. Mirrors
/// the inlined arm, including both request-mismatch violations.
fn apply_generation_finalized(
    generation_finalized_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::GenerationFinalized {
        request_id,
        expert_memory_mode,
        mlx_memory_snapshot,
        expert_residency,
    } = generation_finalized_event
    else {
        return Ok(());
    };
    let chat_request = active_chat_request_mut(active_request)?;
    if request_id != chat_request.request_id {
        return Err(protocol_violation(
            "finalized generation state request mismatch",
        ));
    }
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        chat_request.fold_observed_mlx_peak_memory_bytes(mlx_memory_snapshot.peak_memory_bytes);
        chat_request.last_mlx_active_memory_bytes = Some(mlx_memory_snapshot.active_memory_bytes);
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    } else {
        clear_latest_mlx_memory_snapshot(health_snapshot);
    }
    if let Some(expert_memory_mode) = expert_memory_mode {
        publish_expert_memory_mode(health_snapshot, expert_memory_mode);
    }
    if let Some(expert_residency) = expert_residency {
        chat_request.final_resident_expert_count = Some(expert_residency.resident_expert_count);
        chat_request.final_resident_expert_payload_bytes =
            Some(expert_residency.resident_expert_payload_bytes);
        crate::worker::health::publish_worker_expert_residency(health_snapshot, expert_residency);
    }
    Ok(())
}

/// Applies a generation failure: drops the matching chat request, publishes
/// idle, clears request progress, and delivers the failure to the stream.
/// Mirrors the inlined arm, including all three protocol violations.
fn apply_generation_failed(
    generation_failed_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::Failed { request_id, reason } = generation_failed_event else {
        return Ok(());
    };
    let Some(chat_request) = active_request.as_ref().and_then(ActiveWorkerRequest::chat) else {
        return Err(protocol_violation("failure without an active request"));
    };
    if request_id != chat_request.request_id {
        return Err(protocol_violation("failure request mismatch"));
    }
    let Some(ActiveWorkerRequest::Chat(failed_request)) = active_request.take() else {
        return Err(protocol_violation(
            "chat failure while image generation is active",
        ));
    };
    tracing::warn!(request_id = request_id.value(), failure_reason = ?reason,
        "worker generation failed");
    crate::problem_reports::record_flight_event(crate::problem_reports::FlightEvent::ReplyFailed);
    crate::problem_reports::report_reply_failure(&reason, health_snapshot);
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    try_send_stream_event(
        &failed_request.stream_event_sender,
        ChatGenerationStreamEvent::Failed { reason },
    )
}

/// Publishes the preparation snapshot and delegates the preparation start to
/// the request-scoped handler. Mirrors the inlined arm.
fn apply_generation_preparation_started(
    preparation_started_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::GenerationPreparationStarted {
        request_id,
        expert_residency,
        mlx_memory_snapshot,
    } = preparation_started_event
    else {
        return Ok(());
    };
    if let Some(mlx_memory_snapshot) = &mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, *mlx_memory_snapshot);
    }
    handle_generation_preparation_started(
        request_id,
        expert_residency.map(Into::into),
        health_snapshot,
        active_chat_request_mut(active_request)?,
    )
}

/// Delegates a completed generation to the completion handler. Mirrors the
/// inlined arm.
fn apply_worker_completed(
    worker_completed_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
    completion_log: &mut CompletionAttributionLog,
) -> Result<(), WorkerControlError> {
    // Routed from the dispatch match; total without a silent catch-all.
    let WorkerEvent::Completed {
        request_id,
        prompt_token_count,
        generated_token_count,
        reasoning_token_count,
        astro_ssd_prompt_cache_restored_token_count,
        astro_ssd_prompt_cache_diagnostics,
        reason,
    } = worker_completed_event
    else {
        return Ok(());
    };
    crate::problem_reports::record_flight_event(
        crate::problem_reports::FlightEvent::ReplyFinished {
            prompt_token_count,
            generated_token_count,
        },
    );
    handle_worker_completion_event(
        request_id,
        prompt_token_count,
        generated_token_count,
        reasoning_token_count,
        astro_ssd_prompt_cache_restored_token_count,
        astro_ssd_prompt_cache_diagnostics,
        reason,
        health_snapshot,
        active_request,
        performance_log,
        completion_log,
    )
}

/// Dispatches the request-scoped chat progress events (output fragments,
/// prefill progress, generation progress, first decode, prompt work reuse,
/// drafting report) to their dedicated handlers. `dispatch_chat_generation_event`
/// routes only these events here, so the final arm never runs.
fn dispatch_chat_progress_event(
    worker_progress_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<(), WorkerControlError> {
    match worker_progress_event {
        preparation_started_event @ WorkerEvent::GenerationPreparationStarted { .. } => {
            apply_generation_preparation_started(
                preparation_started_event,
                health_snapshot,
                active_request,
            )
        }
        worker_output_event @ WorkerEvent::Output { .. } => handle_worker_output(
            worker_output_event,
            health_snapshot,
            active_chat_request_mut(active_request)?,
        ),
        worker_prefill_progress_event @ WorkerEvent::PrefillProgress { .. } => {
            handle_worker_prefill_progress(
                worker_prefill_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )
        }
        worker_generation_progress_event @ WorkerEvent::GenerationProgress { .. } => {
            handle_worker_generation_progress(
                worker_generation_progress_event,
                health_snapshot,
                active_chat_request_mut(active_request)?,
            )
        }
        worker_first_decode_event @ WorkerEvent::FirstDecodeCompleted { .. } => {
            handle_worker_first_decode_completed(
                worker_first_decode_event,
                active_chat_request_mut(active_request)?,
            )
        }
        reply_report_event @ (WorkerEvent::PromptWorkReuse { .. }
        | WorkerEvent::DraftingReport { .. }) => handle_worker_reply_report(
            reply_report_event,
            health_snapshot,
            active_chat_request_mut(active_request)?,
        ),
        _ => Ok(()),
    }
}

fn active_chat_request_mut(
    active_request: &mut Option<ActiveWorkerRequest>,
) -> Result<&mut ActiveGeneration, WorkerControlError> {
    active_request
        .as_mut()
        .and_then(ActiveWorkerRequest::chat_mut)
        .ok_or_else(|| protocol_violation("chat event without an active chat request"))
}
