//! The non-command branches of the worker loop's `select!`, extracted from
//! `run_worker`: reacting to a worker-process event, the model-load deadline,
//! the stream/disconnect deadlines, and the idle telemetry tick.
//!
//! Each unit mirrors the inlined branch body it replaced — same error strings,
//! same `is_ready` transitions, same log fields. The pure failure-class
//! decision is kept separate so it can be unit-tested on its own.

use tokio::time::Instant;

use crate::worker::event_handler::handle_worker_event;
use crate::worker::loop_types::{ActiveWorkerRequest, WorkerLoopRuntime};
use crate::worker::process::containment::{cancel_active_generation, contain_worker_failure};
use crate::{ImageGenerationExecutionError, WorkerControlError};

/// How the worker loop reacts to a worker-process event that failed to handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::worker) enum WorkerEventReaction {
    /// The HTTP stream stopped consuming output; cancel the active request but
    /// keep the worker (stream backpressure is recoverable).
    CancelActiveGeneration,
    /// A genuine worker failure: contain it and mark the worker unready.
    ContainWorkerFailure,
}

/// Classifies an event-handling error. Stream backpressure is the only
/// recoverable class; everything else is a worker failure.
pub(in crate::worker) fn worker_event_reaction_to_failure(
    control_error: &WorkerControlError,
) -> WorkerEventReaction {
    if matches!(control_error, WorkerControlError::StreamBackpressure) {
        WorkerEventReaction::CancelActiveGeneration
    } else {
        WorkerEventReaction::ContainWorkerFailure
    }
}

/// Reacts to one worker-process event outcome: maps the failure class to a
/// cancellation or a containment, and marks the worker unready on failure.
/// Mirrors the inlined `worker_process.next_event()` branch.
pub(in crate::worker) async fn react_to_worker_event_outcome(
    runtime: &mut WorkerLoopRuntime,
    worker_event_outcome: Result<
        Option<astronomical_ipc_protocol::WorkerEvent>,
        WorkerControlError,
    >,
) {
    match worker_event_outcome {
        Ok(Some(worker_event)) => {
            if let Err(event_handling_error) = handle_worker_event(
                worker_event,
                &runtime.health_snapshot,
                &mut runtime.is_ready,
                &mut runtime.model_load_deadline,
                &mut runtime.active_generation,
                &mut runtime.performance_log,
                &mut runtime.completion_log,
            ) {
                if worker_event_reaction_to_failure(&event_handling_error)
                    == WorkerEventReaction::CancelActiveGeneration
                {
                    tracing::warn!("HTTP stream stopped consuming output; cancelling request");
                    cancel_active_generation(
                        &mut runtime.worker_process,
                        &runtime.health_snapshot,
                        &mut runtime.active_generation,
                        runtime.cancellation_acknowledgement_timeout,
                        runtime.model_load_timeout,
                        &mut runtime.is_ready,
                    )
                    .await;
                } else {
                    contain_worker_failure(
                        &mut runtime.worker_process,
                        &runtime.health_snapshot,
                        &mut runtime.active_generation,
                        event_handling_error,
                    )
                    .await;
                    runtime.is_ready = false;
                }
            }
        }
        Ok(None) => {
            contain_worker_failure(
                &mut runtime.worker_process,
                &runtime.health_snapshot,
                &mut runtime.active_generation,
                WorkerControlError::WorkerEventStreamClosed,
            )
            .await;
            runtime.is_ready = false;
        }
        Err(worker_event_error) => {
            contain_worker_failure(
                &mut runtime.worker_process,
                &runtime.health_snapshot,
                &mut runtime.active_generation,
                worker_event_error,
            )
            .await;
            runtime.is_ready = false;
        }
    }
}

/// Cancels the active request and marks the worker unready. The four
/// disconnect/deadline branches of the `select!` used to call
/// `cancel_active_generation` with these exact arguments verbatim.
pub(in crate::worker) async fn cancel_active_request(runtime: &mut WorkerLoopRuntime) {
    cancel_active_generation(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
        runtime.cancellation_acknowledgement_timeout,
        runtime.model_load_timeout,
        &mut runtime.is_ready,
    )
    .await;
}

/// Contains a model-load timeout and resets the deadline for the next attempt.
/// Mirrors the inlined `wait_for_deadline(model_load_deadline)` branch,
/// including the reset-only-while-running rule.
pub(in crate::worker) async fn fail_model_load_deadline(runtime: &mut WorkerLoopRuntime) {
    contain_worker_failure(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
        WorkerControlError::ModelLoadTimeout {
            model_load_timeout_millis: runtime.model_load_timeout.as_millis(),
        },
    )
    .await;
    runtime.model_load_deadline = runtime
        .worker_process
        .process_id()
        .map(|_| Instant::now() + runtime.model_load_timeout);
}

/// Sends `DeadlineExceeded` to the active image request's result channel, if
/// the active request is an image. Mirrors the inlined
/// `wait_for_image_execution_deadline` branch's pre-cancel send.
pub(in crate::worker) fn send_image_deadline_exceeded(
    active_generation: &Option<ActiveWorkerRequest>,
) {
    if let Some(ActiveWorkerRequest::Image(active_image)) = active_generation.as_ref() {
        let _send_outcome = active_image
            .image_result_sender
            .try_send(Err(ImageGenerationExecutionError::DeadlineExceeded));
    }
}

/// Requests idle MLX memory telemetry when a model is loaded. Mirrors the
/// inlined idle-telemetry branch body.
pub(in crate::worker) async fn sample_idle_mlx_memory_telemetry(
    runtime: &mut WorkerLoopRuntime,
    has_loaded_model: bool,
) {
    if has_loaded_model
        && let Err(memory_sample_error) = runtime.worker_process.sample_mlx_memory().await
    {
        tracing::warn!(error = %memory_sample_error, "could not request idle MLX memory telemetry");
    }
}
