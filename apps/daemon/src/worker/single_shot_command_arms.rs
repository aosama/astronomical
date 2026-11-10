//! Worker-loop command arms for the requests that run as one forward pass:
//! decisions and embeddings.
//!
//! Both arms do the same three things before their own admission handler
//! runs: refuse while the worker is unavailable, let a pending memory-limit
//! change settle, and contain the worker when admission fails. They live
//! apart from the chat and image arms, whose requests stream progress.

use crate::GenerationStartError;
use crate::worker::command_arms::reject_unavailable_worker;
use crate::worker::decisions_request::handle_generate_decisions_command;
use crate::worker::embeddings_request::handle_generate_embeddings_command;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::worker::memory::limit::take_generation_start_after_pending_memory_limit;
use crate::worker::process::containment::contain_worker_failure;

/// Admits a request that runs as one forward pass: a decisions command
/// through its own arm, and an embeddings command through the other. Any
/// other command is ignored there, as each arm ignores what it does not own.
pub(in crate::worker) async fn admit_single_shot_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    if matches!(routed_command, WorkerLoopCommand::GenerateDecisions { .. }) {
        admit_decisions_generation_command(runtime, is_worker_running, routed_command).await;
        return;
    }
    admit_embeddings_generation_command(runtime, is_worker_running, routed_command).await;
}

/// Handles a decisions generation command, the same way an embeddings one
/// is handled: refuse while the worker is unavailable, let a pending memory
/// limit settle, then admit through the shared single-shot steps.
async fn admit_decisions_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateDecisions {
        active_generation_permit,
        decisions_command,
        start_sender,
        decisions_result_sender,
    } = routed_command
    else {
        return;
    };
    if reject_unavailable_worker(runtime.is_ready, is_worker_running) {
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return;
    }
    let Some(start_sender) = take_generation_start_after_pending_memory_limit(
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.worker_process,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        start_sender,
    )
    .await
    else {
        return;
    };
    if let Err(control_error) = handle_generate_decisions_command(
        &mut runtime.worker_process,
        active_generation_permit,
        decisions_command,
        start_sender,
        decisions_result_sender,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        &runtime.model_policy_catalog,
        runtime.model_load_timeout,
        runtime.cancellation_acknowledgement_timeout,
    )
    .await
    {
        contain_worker_failure(
            &mut runtime.worker_process,
            &runtime.health_snapshot,
            &mut runtime.active_generation,
            control_error,
        )
        .await;
        runtime.is_ready = false;
    }
}

/// Handles an embeddings generation command. Mirrors the inlined
/// `GenerateEmbeddings` arm.
async fn admit_embeddings_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateEmbeddings {
        active_generation_permit,
        embeddings_command,
        start_sender,
        embeddings_result_sender,
        admitted_at,
        queue_wait_elapsed,
    } = routed_command
    else {
        return;
    };
    if reject_unavailable_worker(runtime.is_ready, is_worker_running) {
        let _send_outcome = start_sender.send(Err(GenerationStartError::WorkerUnavailable));
        return;
    }
    let Some(start_sender) = take_generation_start_after_pending_memory_limit(
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.worker_process,
        runtime.model_load_timeout,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        start_sender,
    )
    .await
    else {
        return;
    };
    if let Err(control_error) = handle_generate_embeddings_command(
        &mut runtime.worker_process,
        active_generation_permit,
        embeddings_command,
        start_sender,
        embeddings_result_sender,
        admitted_at,
        queue_wait_elapsed,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        &runtime.model_policy_catalog,
        runtime.model_load_timeout,
        runtime.cancellation_acknowledgement_timeout,
    )
    .await
    {
        contain_worker_failure(
            &mut runtime.worker_process,
            &runtime.health_snapshot,
            &mut runtime.active_generation,
            control_error,
        )
        .await;
        runtime.is_ready = false;
    }
}
