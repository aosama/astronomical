//! Answers a live memory-limit command at the worker loop: queued behind an
//! active generation, or applied at once when idle, with any failure of the
//! idle application contained so the failure relaunch can recover the worker.

use crate::WorkerControlError;
use crate::worker::health::publish_pending_mlx_memory_ceiling;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::worker::memory::limit::{
    MlxMemoryLimitUpdateOutcome, PendingMlxMemoryLimitUpdate, apply_mlx_memory_limit,
    contain_mlx_memory_limit_failure,
};

/// Queues the MLX memory-limit update while a generation is active, or applies
/// it immediately when idle. Mirrors the inlined `UpdateMlxMemoryLimit` arm,
/// including the pending-ceiling health publish on the queued path.
pub(in crate::worker) async fn queue_or_apply_mlx_memory_limit_update(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::UpdateMlxMemoryLimit {
        effective_mlx_memory_ceiling_bytes,
        configuration_generation,
        update_sender,
    } = routed_command
    else {
        return;
    };
    if runtime.active_generation.is_some() {
        runtime.pending_mlx_memory_limit_update = Some(PendingMlxMemoryLimitUpdate {
            effective_mlx_memory_ceiling_bytes,
            configuration_generation,
        });
        publish_pending_mlx_memory_ceiling(
            &runtime.health_snapshot,
            Some(effective_mlx_memory_ceiling_bytes),
        );
        let _send_outcome = update_sender.send(Ok(MlxMemoryLimitUpdateOutcome::Queued));
        return;
    }
    let update_outcome = apply_mlx_memory_limit(
        &mut runtime.worker_process,
        effective_mlx_memory_ceiling_bytes,
        configuration_generation,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
    )
    .await;
    let update_outcome = contain_failed_idle_memory_limit_update(runtime, update_outcome).await;
    let _send_outcome = update_sender.send(update_outcome);
}

/// A worker that fails while adopting an idle ceiling (it may have exited, as
/// an image engine does on a fatal planning error) must go through the same
/// containment as every other worker failure. Without it the dead process is
/// never marked stopped and unavailable, health keeps reporting the old ready
/// model, and the #1801 relaunch never fires, so every later request is
/// refused until the app restarts (#1900). The queued and drained paths
/// already contain; this was the one idle path that did not.
/// Applied, queued, and rejected outcomes pass through untouched.
async fn contain_failed_idle_memory_limit_update(
    runtime: &mut WorkerLoopRuntime,
    update_outcome: Result<MlxMemoryLimitUpdateOutcome, WorkerControlError>,
) -> Result<MlxMemoryLimitUpdateOutcome, WorkerControlError> {
    let Err(memory_limit_error) = update_outcome else {
        return update_outcome;
    };
    if matches!(
        memory_limit_error,
        WorkerControlError::MlxMemoryLimitUpdateTimeout { .. }
    ) {
        // apply_mlx_memory_limit already contained the timed-out worker, and
        // callers report this timeout by its own message.
        return Err(memory_limit_error);
    }
    let failure_description = memory_limit_error.to_string();
    contain_mlx_memory_limit_failure(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
        &mut runtime.is_ready,
        memory_limit_error,
    )
    .await;
    Err(WorkerControlError::ContainedMlxMemoryLimitFailure {
        failure_description,
    })
}
