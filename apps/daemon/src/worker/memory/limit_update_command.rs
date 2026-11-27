//! Answers a live memory-limit command at the worker loop: applied at once,
//! during a reply or not (issue #2681), with any failure contained so the
//! failure relaunch can recover the worker.

use crate::WorkerControlError;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::worker::memory::limit::{
    MlxMemoryLimitUpdateOutcome, apply_mlx_memory_limit, contain_mlx_memory_limit_failure,
};

/// Sends the MLX memory-limit update to the worker straight away, even while
/// a request runs: the worker moves the limit between two of its steps, or
/// refuses it with the memory the model needs, and the request goes on.
pub(in crate::worker) async fn apply_mlx_memory_limit_update(
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
    let update_outcome = apply_mlx_memory_limit(
        runtime,
        effective_mlx_memory_ceiling_bytes,
        configuration_generation,
    )
    .await;
    let update_outcome = contain_failed_memory_limit_update(runtime, update_outcome).await;
    let _send_outcome = update_sender.send(update_outcome);
}

/// A worker that fails while adopting a ceiling (it may have exited, as an
/// image engine does on a fatal planning error) must go through the same
/// containment as every other worker failure. Without it the dead process is
/// never marked stopped and unavailable, health keeps reporting the old ready
/// model, and the #1801 relaunch never fires, so every later request is
/// refused until the app restarts (#1900). Applied and rejected outcomes pass
/// through untouched.
async fn contain_failed_memory_limit_update(
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
