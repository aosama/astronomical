use std::{
    sync::{Arc, RwLock},
    time::Duration,
};

use ipc_protocol::WorkerEvent;
use tokio::time::timeout;

use crate::worker::event_handler;
use crate::worker::event_reaction::cancel_active_request;
use crate::worker::loop_types::{ActiveWorkerRequest, WorkerLoopRuntime};
use crate::worker::process::containment;
use crate::{WorkerControlError, WorkerHealthSnapshot, WorkerProcess};

/// Completion state returned by a live MLX memory-limit request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxMemoryLimitUpdateOutcome {
    Applied,
    Rejected,
}

/// Sends a live ceiling to the worker and waits for its answer, during a
/// reply or not (issue #2681).
///
/// The worker answers between two steps of a running request, so the wait
/// keeps handling that request's events: a reply's text still reaches its
/// client while the limit moves. A client that stops reading during the wait
/// has its request cancelled once the worker has answered, as the worker loop
/// would have done, so the answer itself is never lost.
pub(in crate::worker) async fn apply_mlx_memory_limit(
    runtime: &mut WorkerLoopRuntime,
    effective_mlx_memory_ceiling_bytes: u64,
    configuration_generation: String,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    let memory_limit_update_timeout = runtime.model_load_timeout;
    let configuration_generation_for_relaunch = configuration_generation.clone();
    let mut reply_stream_stopped_consuming = false;
    let update_outcome = timeout(
        memory_limit_update_timeout,
        send_and_wait_for_memory_limit_answer(
            runtime,
            effective_mlx_memory_ceiling_bytes,
            configuration_generation,
            &mut reply_stream_stopped_consuming,
        ),
    )
    .await;
    let Ok(memory_limit_update_outcome) = update_outcome else {
        return Err(
            contain_timed_out_memory_limit_update(runtime, memory_limit_update_timeout).await,
        );
    };
    if reply_stream_stopped_consuming {
        tracing::warn!("HTTP stream stopped consuming output; cancelling request");
        cancel_active_request(runtime).await;
    }
    adopt_applied_ceiling_into_relaunch_snapshot(
        &mut runtime.worker_process,
        memory_limit_update_outcome,
        effective_mlx_memory_ceiling_bytes,
        &configuration_generation_for_relaunch,
    )
}

async fn send_and_wait_for_memory_limit_answer(
    runtime: &mut WorkerLoopRuntime,
    effective_mlx_memory_ceiling_bytes: u64,
    configuration_generation: String,
    reply_stream_stopped_consuming: &mut bool,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    runtime
        .worker_process
        .update_mlx_memory_limit(effective_mlx_memory_ceiling_bytes, configuration_generation)
        .await?;
    loop {
        let worker_event = runtime
            .worker_process
            .next_event()
            .await?
            .ok_or(WorkerControlError::WorkerEventStreamClosed)?;
        let memory_limit_update_outcome = memory_limit_answer(&worker_event);
        match event_handler::handle_worker_event(
            worker_event,
            &runtime.health_snapshot,
            &mut runtime.is_ready,
            &mut runtime.model_load_deadline,
            &mut runtime.active_generation,
            &mut runtime.performance_log,
            &mut runtime.completion_log,
        ) {
            Ok(()) => {}
            Err(WorkerControlError::StreamBackpressure) => *reply_stream_stopped_consuming = true,
            Err(event_handling_error) => return Err(event_handling_error),
        }
        if let Some(memory_limit_update_outcome) = memory_limit_update_outcome {
            return Ok(memory_limit_update_outcome);
        }
    }
}

/// The worker's answer to a live ceiling, or `None` for any other event.
fn memory_limit_answer(worker_event: &WorkerEvent) -> Option<MlxMemoryLimitUpdateOutcome> {
    match worker_event {
        WorkerEvent::MlxMemoryLimitChanged { .. } => Some(MlxMemoryLimitUpdateOutcome::Applied),
        WorkerEvent::MlxMemoryLimitRejected { .. } => Some(MlxMemoryLimitUpdateOutcome::Rejected),
        _ => None,
    }
}

async fn contain_timed_out_memory_limit_update(
    runtime: &mut WorkerLoopRuntime,
    memory_limit_update_timeout: Duration,
) -> WorkerControlError {
    let memory_limit_update_timeout_millis = memory_limit_update_timeout.as_millis();
    containment::contain_worker_failure(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
        WorkerControlError::MlxMemoryLimitUpdateTimeout {
            memory_limit_update_timeout_millis,
        },
    )
    .await;
    runtime.is_ready = false;
    WorkerControlError::MlxMemoryLimitUpdateTimeout {
        memory_limit_update_timeout_millis,
    }
}

/// Stores the requested ceiling on the launch snapshot only after the worker
/// adopts it. A rejected or failed update must leave the snapshot that the
/// next process boots from unchanged.
fn adopt_applied_ceiling_into_relaunch_snapshot(
    worker_process: &mut WorkerProcess,
    memory_limit_update_outcome: Result<MlxMemoryLimitUpdateOutcome, WorkerControlError>,
    effective_mlx_memory_ceiling_bytes: u64,
    configuration_generation: &str,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    if memory_limit_outcome_was_applied(&memory_limit_update_outcome) {
        worker_process.record_applied_memory_ceiling_for_relaunch(
            effective_mlx_memory_ceiling_bytes,
            configuration_generation,
        );
    }
    memory_limit_update_outcome
}

/// Reports whether this update adopted a ceiling the running worker will keep.
///
/// A rejection or a control error must not overwrite the ceiling the next
/// process boots from.
fn memory_limit_outcome_was_applied(
    memory_limit_update_outcome: &Result<MlxMemoryLimitUpdateOutcome, WorkerControlError>,
) -> bool {
    matches!(
        memory_limit_update_outcome,
        Ok(MlxMemoryLimitUpdateOutcome::Applied)
    )
}

pub(in crate::worker) async fn contain_mlx_memory_limit_failure(
    worker_process: &mut WorkerProcess,
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    active_request: &mut Option<ActiveWorkerRequest>,
    is_ready: &mut bool,
    memory_limit_error: WorkerControlError,
) {
    if !matches!(
        &memory_limit_error,
        WorkerControlError::MlxMemoryLimitUpdateTimeout { .. }
    ) {
        containment::contain_worker_failure(
            worker_process,
            health_snapshot,
            active_request,
            memory_limit_error,
        )
        .await;
    }
    *is_ready = false;
}

#[cfg(test)]
#[path = "limit_outcome_tests.rs"]
mod limit_outcome_tests;
