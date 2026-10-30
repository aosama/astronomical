//! Correlates image events and releases shared ownership only after cleanup telemetry.

use std::sync::{Arc, RwLock};

use astronomical_ipc_protocol::{MlxMemorySnapshotSource, WorkerEvent, WorkerMlxMemorySnapshot};

use crate::{
    ActiveRequestProgress, GenerationPerformanceLog, ImageGenerationExecutionError,
    ImageGenerationOutput, WorkerActivity, WorkerControlError, WorkerHealthSnapshot,
    performance::generation_log::{ImageGenerationPerformanceRecord, unix_epoch_millis},
    worker::event_handler::protocol_violation,
    worker::health::{
        clear_active_request_progress, clear_latest_mlx_memory_snapshot,
        publish_active_request_progress, publish_activity, publish_latest_mlx_memory_snapshot,
    },
    worker::loop_types::{ActiveImageGeneration, ActiveWorkerRequest},
};

pub(super) fn handle_worker_image_event(
    worker_event: WorkerEvent,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    performance_log: &mut GenerationPerformanceLog,
) -> Result<(), WorkerControlError> {
    match worker_event {
        WorkerEvent::ImageGenerationProgress {
            request_id,
            phase,
            completed_steps,
            total_steps,
            elapsed_millis,
            mlx_memory_snapshot,
        } => apply_image_generation_progress(
            active_worker_request,
            health_snapshot,
            request_id,
            phase,
            completed_steps,
            total_steps,
            elapsed_millis,
            mlx_memory_snapshot,
        ),
        WorkerEvent::ImageGenerationCompleted {
            request_id,
            generated_image,
            result_metadata,
        } => apply_image_generation_completed(
            active_worker_request,
            request_id,
            generated_image,
            result_metadata,
        ),
        WorkerEvent::ImageGenerationFailed { request_id, reason } => {
            apply_image_generation_failed(active_worker_request, request_id, reason)
        }
        WorkerEvent::ImageGenerationFinalized {
            request_id,
            elapsed_millis: worker_reported_elapsed_millis,
            mlx_memory_snapshot,
        } => apply_image_generation_finalized(
            active_worker_request,
            health_snapshot,
            performance_log,
            request_id,
            worker_reported_elapsed_millis,
            mlx_memory_snapshot,
        ),
        _ => Err(protocol_violation(
            "non-image event passed to image handler",
        )),
    }
}

/// Applies one in-render progress event: regression-checked field updates
/// followed by the health-surface republication the menu renders from.
#[allow(clippy::too_many_arguments)]
fn apply_image_generation_progress(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    request_id: astronomical_ipc_protocol::RequestId,
    phase: astronomical_ipc_protocol::ImageGenerationPhase,
    completed_steps: u16,
    total_steps: u16,
    elapsed_millis: u64,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) -> Result<(), WorkerControlError> {
    let active_image = matching_active_image(active_worker_request, request_id)?;
    let progress_advanced = active_image.latest_phase != Some(phase)
        || completed_steps > active_image.latest_completed_steps
        || elapsed_millis > active_image.latest_elapsed_millis;
    validate_image_generation_progress(
        active_image,
        phase,
        completed_steps,
        total_steps,
        elapsed_millis,
    )?;
    active_image.latest_phase = Some(phase);
    active_image.latest_completed_steps = completed_steps;
    active_image.latest_elapsed_millis = elapsed_millis;
    if progress_advanced {
        active_image.progress_stall_deadline =
            tokio::time::Instant::now() + active_image.progress_stall_timeout;
    }
    publish_activity(health_snapshot, WorkerActivity::ImageGeneration);
    publish_active_request_progress(
        health_snapshot,
        ActiveRequestProgress::ImageGeneration {
            phase,
            completed_steps,
            total_steps,
            elapsed_millis,
        },
    );
    // Mid-render samples stay unvalidated like text-path progress snapshots; only the
    // finalization snapshot proves cleanup, so a step sample never blocks the render.
    if let Some(mlx_memory_snapshot) = mlx_memory_snapshot {
        publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    }
    Ok(())
}

/// Rejects progress that moves backwards in steps, elapsed time, or phase
/// rank, or that arrives once a terminal outcome is already recorded. The
/// guard expression and its operand order are the original inline check.
fn validate_image_generation_progress(
    active_image: &ActiveImageGeneration,
    phase: astronomical_ipc_protocol::ImageGenerationPhase,
    completed_steps: u16,
    total_steps: u16,
    elapsed_millis: u64,
) -> Result<(), WorkerControlError> {
    if active_image.terminal_outcome.is_some()
        || total_steps != active_image.settings.steps
        || completed_steps < active_image.latest_completed_steps
        || completed_steps > total_steps
        || elapsed_millis < active_image.latest_elapsed_millis
        || active_image
            .latest_phase
            .is_some_and(|latest_phase| image_phase_rank(phase) < image_phase_rank(latest_phase))
    {
        return Err(protocol_violation("invalid image generation progress"));
    }
    Ok(())
}

/// Records a completed image as the request's terminal outcome.
///
/// The encoded bytes stay private until finalization proves request arrays
/// and allocator cache were released, so a cleanup failure cannot let a
/// partial success escape.
fn apply_image_generation_completed(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: astronomical_ipc_protocol::RequestId,
    generated_image: astronomical_ipc_protocol::GeneratedImage,
    result_metadata: astronomical_ipc_protocol::ImageGenerationResultMetadata,
) -> Result<(), WorkerControlError> {
    let active_image = matching_active_image(active_worker_request, request_id)?;
    validate_image_completed_result(active_image, &generated_image, &result_metadata)?;
    // Encoded bytes remain private until finalization proves request arrays and allocator
    // cache were released; this prevents partial success from escaping on cleanup failure.
    active_image.terminal_outcome = Some(Ok(ImageGenerationOutput {
        generated_image,
        result_metadata,
    }));
    active_image.latest_elapsed_millis = result_metadata.elapsed_millis;
    active_image.terminal_received_at = Some(tokio::time::Instant::now());
    Ok(())
}

/// Validates one completed image against the admitted request settings and
/// the deliverable-image contract. The guard expression and its operand
/// order are the original inline check.
fn validate_image_completed_result(
    active_image: &ActiveImageGeneration,
    generated_image: &astronomical_ipc_protocol::GeneratedImage,
    result_metadata: &astronomical_ipc_protocol::ImageGenerationResultMetadata,
) -> Result<(), WorkerControlError> {
    if active_image.terminal_outcome.is_some()
        || result_metadata.width_pixels != active_image.settings.width_pixels
        || result_metadata.height_pixels != active_image.settings.height_pixels
        || result_metadata.steps != active_image.settings.steps
        || result_metadata.guidance_thousandths != active_image.settings.guidance_thousandths
        || result_metadata.seed != active_image.settings.seed
        || result_metadata.elapsed_millis < active_image.latest_elapsed_millis
        || generated_image.mime_type != "image/png"
        || generated_image.encoded_bytes.is_empty()
    {
        return Err(protocol_violation("invalid completed image result"));
    }
    Ok(())
}

/// Records a request-scoped failure as the request's terminal outcome.
fn apply_image_generation_failed(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: astronomical_ipc_protocol::RequestId,
    reason: astronomical_ipc_protocol::ImageGenerationFailureReason,
) -> Result<(), WorkerControlError> {
    let active_image = matching_active_image(active_worker_request, request_id)?;
    if active_image.terminal_outcome.is_some() {
        return Err(protocol_violation("duplicate image terminal outcome"));
    }
    active_image.terminal_outcome = Some(Err(reason));
    active_image.terminal_received_at = Some(tokio::time::Instant::now());
    Ok(())
}

/// Releases the active image request: publishes the finalization memory
/// snapshot, records the performance record, and only then hands the
/// terminal outcome to the awaiting result channel.
fn apply_image_generation_finalized(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    performance_log: &mut GenerationPerformanceLog,
    request_id: astronomical_ipc_protocol::RequestId,
    worker_reported_elapsed_millis: u64,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) -> Result<(), WorkerControlError> {
    let active_image = matching_active_image(active_worker_request, request_id)?;
    if worker_reported_elapsed_millis < active_image.latest_elapsed_millis {
        return Err(protocol_violation(
            "image finalization elapsed time regressed",
        ));
    }
    let terminal_outcome = active_image
        .terminal_outcome
        .take()
        .ok_or_else(|| protocol_violation("image finalized before a terminal outcome"))?;
    let mlx_peak_memory_bytes = mlx_memory_snapshot
        .as_ref()
        .map(|snapshot| snapshot.peak_memory_bytes);
    let mlx_active_memory_bytes = mlx_memory_snapshot
        .as_ref()
        .map(|snapshot| snapshot.active_memory_bytes);
    publish_image_finalized_memory_snapshot(health_snapshot, mlx_memory_snapshot)?;
    let Some(ActiveWorkerRequest::Image(finalized_image)) = active_worker_request.take() else {
        return Err(protocol_violation(
            "image finalization lost active ownership",
        ));
    };
    publish_activity(health_snapshot, WorkerActivity::Idle);
    clear_active_request_progress(health_snapshot);
    let completion_outcome = if terminal_outcome.is_ok() {
        "completed"
    } else {
        "failed"
    };
    let encoded_image_bytes = terminal_outcome
        .as_ref()
        .ok()
        .and_then(|output| output.generated_image.encoded_bytes.len().try_into().ok());
    let finalized_at = tokio::time::Instant::now();
    let finalization_durations = image_finalization_durations(&finalized_image, finalized_at)?;
    performance_log.record_image(&ImageGenerationPerformanceRecord {
        operation: "image_generation",
        timestamp_millis: unix_epoch_millis(),
        request_id: request_id.value(),
        model_id: finalized_image.model_id,
        width_pixels: finalized_image.settings.width_pixels,
        height_pixels: finalized_image.settings.height_pixels,
        steps: finalized_image.settings.steps,
        completion_outcome,
        total_elapsed_millis: finalization_durations.total_elapsed_millis,
        queue_wait_elapsed_millis: finalization_durations.queue_wait_elapsed_millis,
        swap_load_elapsed_millis: finalization_durations.swap_load_elapsed_millis,
        execution_elapsed_millis: finalization_durations.execution_elapsed_millis,
        finalization_elapsed_millis: finalization_durations.finalization_elapsed_millis,
        worker_reported_elapsed_millis,
        encoded_image_bytes,
        mlx_peak_memory_bytes,
        mlx_active_memory_bytes,
    });
    let public_outcome = terminal_outcome.map_err(ImageGenerationExecutionError::WorkerFailure);
    match finalized_image.image_result_sender.try_send(public_outcome) {
        Ok(()) | Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {}
        Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
            return Err(WorkerControlError::StreamBackpressure);
        }
    }
    Ok(())
}

/// The performance-record duration windows measured from one admitted
/// request's lifecycle timestamps.
#[derive(Debug)]
struct ImageFinalizationDurations {
    total_elapsed_millis: u64,
    queue_wait_elapsed_millis: u64,
    swap_load_elapsed_millis: u64,
    execution_elapsed_millis: u64,
    finalization_elapsed_millis: u64,
}

/// Measures the performance-record duration windows from the admitted
/// request's timestamps. `terminal_received_at` anchors both the execution
/// and cleanup windows, so its absence is a protocol violation detected
/// before any record is written.
fn image_finalization_durations(
    finalized_image: &ActiveImageGeneration,
    finalized_at: tokio::time::Instant,
) -> Result<ImageFinalizationDurations, WorkerControlError> {
    let total_elapsed_millis =
        duration_millis(finalized_at.duration_since(finalized_image.admitted_at));
    let queue_wait_elapsed_millis = duration_millis(finalized_image.queue_wait_elapsed);
    let swap_load_elapsed_millis = duration_millis(finalized_image.swap_load_elapsed);
    let execution_elapsed_millis = finalized_image
        .terminal_received_at
        .map(|terminal_received_at| {
            duration_millis(
                terminal_received_at.duration_since(finalized_image.execution_started_at),
            )
        })
        .ok_or_else(|| protocol_violation("image finalization lost terminal timing"))?;
    let finalization_elapsed_millis = finalized_image
        .terminal_received_at
        .map(|terminal_received_at| {
            duration_millis(finalized_at.duration_since(terminal_received_at))
        })
        .ok_or_else(|| protocol_violation("image finalization lost cleanup timing"))?;
    Ok(ImageFinalizationDurations {
        total_elapsed_millis,
        queue_wait_elapsed_millis,
        swap_load_elapsed_millis,
        execution_elapsed_millis,
        finalization_elapsed_millis,
    })
}

fn image_phase_rank(phase: astronomical_ipc_protocol::ImageGenerationPhase) -> u8 {
    use astronomical_ipc_protocol::ImageGenerationPhase;
    match phase {
        ImageGenerationPhase::Preparing => 0,
        ImageGenerationPhase::EncodingPrompt => 1,
        ImageGenerationPhase::Denoising => 2,
        ImageGenerationPhase::Decoding => 3,
        ImageGenerationPhase::EncodingImage => 4,
    }
}

fn duration_millis(duration: std::time::Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

pub(super) fn publish_image_finalized_memory_snapshot(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    mlx_memory_snapshot: Option<WorkerMlxMemorySnapshot>,
) -> Result<(), WorkerControlError> {
    let Some(mlx_memory_snapshot) = mlx_memory_snapshot else {
        clear_latest_mlx_memory_snapshot(health_snapshot);
        return Ok(());
    };
    let attributed_memory_bytes = mlx_memory_snapshot
        .expert_payload_bytes
        .saturating_add(mlx_memory_snapshot.model_core_payload_bytes)
        .saturating_add(mlx_memory_snapshot.context_state_payload_bytes);
    let effective_ceiling_bytes = health_snapshot
        .read()
        .ok()
        .map(|snapshot| snapshot.mlx_memory_ceiling_bytes)
        .unwrap_or(0);
    if mlx_memory_snapshot.source != MlxMemorySnapshotSource::Finalized
        || mlx_memory_snapshot.allocator_cache_memory_bytes != 0
        || mlx_memory_snapshot.active_memory_bytes > mlx_memory_snapshot.peak_memory_bytes
        || attributed_memory_bytes > mlx_memory_snapshot.active_memory_bytes
        || mlx_memory_snapshot.expert_payload_bytes != 0
        || mlx_memory_snapshot.context_state_payload_bytes != 0
        || (effective_ceiling_bytes > 0
            && mlx_memory_snapshot.active_memory_bytes > effective_ceiling_bytes)
    {
        return Err(protocol_violation(
            "invalid image finalization MLX memory snapshot",
        ));
    }
    publish_latest_mlx_memory_snapshot(health_snapshot, mlx_memory_snapshot);
    Ok(())
}

fn matching_active_image(
    active_worker_request: &mut Option<ActiveWorkerRequest>,
    request_id: astronomical_ipc_protocol::RequestId,
) -> Result<&mut ActiveImageGeneration, WorkerControlError> {
    let active_image = active_worker_request
        .as_mut()
        .and_then(ActiveWorkerRequest::image_mut)
        .ok_or_else(|| protocol_violation("image event without an active image request"))?;
    if active_image.request_id != request_id {
        return Err(protocol_violation("image event request mismatch"));
    }
    Ok(active_image)
}

#[cfg(test)]
#[path = "image_event_tests.rs"]
mod image_event_tests;
