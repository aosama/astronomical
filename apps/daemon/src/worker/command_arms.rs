//! The worker loop's iteration frame and per-command glue, extracted from
//! `run_worker`.
//!
//! `run_worker`'s `tokio::select!` used to inline one large `match` over
//! `WorkerLoopCommand`, with each arm repeating the same four steps: the
//! readiness guard, the pending-memory-limit gate, delegation to the
//! modality handler, and failure containment. This module makes the
//! iteration frame and each arm a named unit, all taking the loop-carried
//! `WorkerLoopRuntime` by `&mut` so no unit needs a dozen individual
//! references.
//!
//! Every unit mirrors the inlined body it replaced expression-for-expression:
//! the same guards, the same call order, the same error strings and log
//! fields, and the same `is_ready = false` transitions.

use tokio::sync::mpsc;
use tokio::time::Interval;

use crate::serving::chat_generation_executor::{wait_for_deadline, wait_for_stream_disconnect};
use crate::worker::cache_clear::handle_prompt_cache_clear_command;
use crate::worker::embeddings_request::handle_generate_embeddings_command;
use crate::worker::event_reaction::{
    cancel_active_request, fail_model_load_deadline, react_to_worker_event_outcome,
    sample_idle_mlx_memory_telemetry, send_image_deadline_exceeded,
};
use crate::worker::generation::generate::handle_generate_command;
use crate::worker::health::publish_pending_mlx_memory_ceiling;
use crate::worker::idle_drain::{
    drain_pending_idle_work, wait_for_embeddings_execution_deadline,
    wait_for_image_execution_deadline,
};
use crate::worker::image_request::handle_generate_image_command;
use crate::worker::loop_types::{
    ActiveWorkerRequest, WorkerLoopCommand, WorkerLoopRuntime, WorkerLoopState,
};
use crate::worker::memory::demotion::MemoryDemotionRestart;
use crate::worker::memory::limit::{
    MlxMemoryLimitUpdateOutcome, PendingMlxMemoryLimitUpdate, apply_mlx_memory_limit,
    take_generation_start_after_pending_memory_limit,
};
use crate::worker::process::containment::{
    close_worker_if_running, contain_worker_failure, shutdown_worker,
};
use crate::worker::process::replacement::WorkerReplacement;
use crate::{
    ChatGenerationStreamEvent, EmbeddingsExecutionError, EmbeddingsOutput, GenerationStartError,
    ImageGenerationExecutionError, ImageGenerationOutput, WorkerControlError,
};

/// How the worker loop should proceed after one iteration or command arm.
///
/// `Continue` is the ordinary case — the arm handled the command and the loop
/// re-arms its `select!`. `Exit` is returned only when the loop must stop for
/// good: the command channel closed or the shutdown command arrived.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::worker) enum WorkerLoopFlow {
    Continue,
    Exit,
}

/// Per-iteration inputs for the worker loop's `select!`: one channel handle
/// per modality-aware branch (cloned from the active request), whether the
/// worker process is running, and whether a model is loaded.
pub(in crate::worker) struct WorkerLoopIterationInputs {
    pub chat_stream_event_sender: Option<mpsc::Sender<ChatGenerationStreamEvent>>,
    pub image_result_sender:
        Option<mpsc::Sender<Result<ImageGenerationOutput, ImageGenerationExecutionError>>>,
    pub embeddings_result_sender:
        Option<mpsc::Sender<Result<EmbeddingsOutput, EmbeddingsExecutionError>>>,
    pub is_worker_running: bool,
    pub has_loaded_model: bool,
}

/// Builds the per-iteration `select!` inputs after the idle-drain pass.
/// Mirrors the sender/flag extraction the loop used to inline: clone each
/// active request's channel, ask the process for liveness, and read the
/// loaded-model id out of the health snapshot.
pub(in crate::worker) fn prepare_worker_loop_iteration(
    runtime: &WorkerLoopRuntime,
) -> WorkerLoopIterationInputs {
    WorkerLoopIterationInputs {
        chat_stream_event_sender: runtime
            .active_generation
            .as_ref()
            .and_then(ActiveWorkerRequest::chat_stream_event_sender),
        image_result_sender: runtime
            .active_generation
            .as_ref()
            .and_then(ActiveWorkerRequest::image_result_sender),
        embeddings_result_sender: runtime
            .active_generation
            .as_ref()
            .and_then(ActiveWorkerRequest::embeddings_result_sender),
        is_worker_running: runtime.worker_process.process_id().is_some(),
        has_loaded_model: runtime
            .health_snapshot
            .read()
            .ok()
            .is_some_and(|worker_health_snapshot| worker_health_snapshot.ready_model_id.is_some()),
    }
}

/// Runs one iteration of the worker event loop: the idle-drain pass, the
/// per-iteration channel extraction, and the `select!` whose command branch
/// dispatches through `handle_worker_loop_command` and whose event branches
/// delegate to `event_reaction`. Returns `Exit` only when the loop must stop
/// (command channel closed or shutdown handled).
pub(in crate::worker) async fn run_worker_iteration(
    runtime: &mut WorkerLoopRuntime,
    command_receiver: &mut mpsc::Receiver<WorkerLoopCommand>,
    idle_control_interval: &mut Interval,
) -> WorkerLoopFlow {
    let mut worker_loop_state = WorkerLoopState {
        worker_process: &mut runtime.worker_process,
        health_snapshot: &runtime.health_snapshot,
        is_ready: &mut runtime.is_ready,
        model_load_deadline: &mut runtime.model_load_deadline,
        active_request: &mut runtime.active_generation,
        performance_log: &mut runtime.performance_log,
        completion_log: &mut runtime.completion_log,
        model_load_timeout: runtime.model_load_timeout,
    };
    drain_pending_idle_work(
        &mut worker_loop_state,
        &mut runtime.pending_mlx_memory_limit_update,
        &mut runtime.pending_prompt_cache_clear,
        &runtime.active_generation_permits,
        &runtime.generation_queue_permits,
    )
    .await;
    let WorkerLoopIterationInputs {
        chat_stream_event_sender,
        image_result_sender,
        embeddings_result_sender,
        is_worker_running,
        has_loaded_model,
    } = prepare_worker_loop_iteration(runtime);

    tokio::select! {
        worker_loop_command = command_receiver.recv() => {
            let Some(worker_loop_command) = worker_loop_command else {
                close_worker_if_running(&mut runtime.worker_process).await;
                return WorkerLoopFlow::Exit;
            };
            match handle_worker_loop_command(runtime, is_worker_running, worker_loop_command).await {
                WorkerLoopFlow::Exit => return WorkerLoopFlow::Exit,
                WorkerLoopFlow::Continue => {}
            }
        }
        worker_event_outcome = runtime.worker_process.next_event(), if is_worker_running => {
            react_to_worker_event_outcome(runtime, worker_event_outcome).await;
        }
        () = wait_for_deadline(runtime.model_load_deadline), if is_worker_running && !runtime.is_ready => {
            fail_model_load_deadline(runtime).await;
        }
        () = wait_for_stream_disconnect(chat_stream_event_sender), if runtime.active_generation.is_some() => {
            cancel_active_request(runtime).await;
        }
        () = crate::serving::image_generation_executor::wait_for_image_disconnect(image_result_sender),
            if matches!(runtime.active_generation, Some(ActiveWorkerRequest::Image(_))) => {
            cancel_active_request(runtime).await;
        }
        () = crate::serving::embeddings_executor::wait_for_embeddings_disconnect(embeddings_result_sender),
            if matches!(runtime.active_generation, Some(ActiveWorkerRequest::Embeddings(_))) => {
            cancel_active_request(runtime).await;
        }
        () = wait_for_embeddings_execution_deadline(&runtime.active_generation),
            if matches!(runtime.active_generation, Some(ActiveWorkerRequest::Embeddings(_))) => {
            cancel_active_request(runtime).await;
        }
        () = wait_for_image_execution_deadline(&runtime.active_generation),
            if matches!(runtime.active_generation, Some(ActiveWorkerRequest::Image(_))) => {
            send_image_deadline_exceeded(&runtime.active_generation);
            cancel_active_request(runtime).await;
        }
        _ = idle_control_interval.tick(), if is_worker_running
            && runtime.is_ready
            && runtime.active_generation.is_none()
            && (has_loaded_model || runtime.pending_prompt_cache_clear.is_some()) => {
            sample_idle_mlx_memory_telemetry(runtime, has_loaded_model).await;
        }
    }
    WorkerLoopFlow::Continue
}

/// Dispatches one worker-loop command to the matching arm unit. The `match`
/// here is the only place that knows which arm handles which variant; the arms
/// themselves are the behavior. Each arm receives the whole command and
/// destructures it internally, which keeps this dispatch one line per variant.
pub(in crate::worker) async fn handle_worker_loop_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    worker_loop_command: WorkerLoopCommand,
) -> WorkerLoopFlow {
    match worker_loop_command {
        routed_command @ WorkerLoopCommand::Generate { .. } => {
            admit_chat_generation_command(runtime, is_worker_running, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::GenerateImage { .. } => {
            admit_image_generation_command(runtime, is_worker_running, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::GenerateEmbeddings { .. } => {
            admit_embeddings_generation_command(runtime, is_worker_running, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::Shutdown { .. } => {
            handle_shutdown_command(runtime, routed_command).await
        }
        routed_command @ WorkerLoopCommand::RestartWorker { .. } => {
            execute_worker_replacement(runtime, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::UpdateMlxMemoryLimit { .. } => {
            queue_or_apply_mlx_memory_limit_update(runtime, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::RestartWorkerForMemoryDemotion { .. } => {
            execute_memory_demotion_restart(runtime, routed_command).await;
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::UpdateModelPolicyCatalog { .. } => {
            replace_model_policy_catalog(runtime, routed_command);
            WorkerLoopFlow::Continue
        }
        routed_command @ WorkerLoopCommand::ClearPromptCache { .. } => {
            clear_prompt_cache(runtime, routed_command).await;
            WorkerLoopFlow::Continue
        }
    }
}

/// The readiness guard shared by the three generation arms: a worker that is
/// not ready, or not running, rejects the request with
/// `WorkerUnavailable` and the arm stops before any admission work.
pub(in crate::worker) fn reject_unavailable_worker(
    is_ready: bool,
    is_worker_running: bool,
) -> bool {
    !is_ready || !is_worker_running
}

/// Handles a chat generation command: readiness guard, pending-memory-limit
/// gate, delegation to the chat handler, and failure containment. Mirrors the
/// inlined `Generate` arm. Returns early (as the original `continue`d) when the
/// guard rejects or the gate declines the request.
async fn admit_chat_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    // The dispatch match routes only Generate commands here; the let-else
    // keeps that contract total without a catch-all arm.
    let WorkerLoopCommand::Generate {
        active_generation_permit,
        generation_command,
        start_sender,
        stream_event_sender,
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
    let mut worker_loop_state = WorkerLoopState {
        worker_process: &mut runtime.worker_process,
        health_snapshot: &runtime.health_snapshot,
        is_ready: &mut runtime.is_ready,
        model_load_deadline: &mut runtime.model_load_deadline,
        active_request: &mut runtime.active_generation,
        performance_log: &mut runtime.performance_log,
        completion_log: &mut runtime.completion_log,
        model_load_timeout: runtime.model_load_timeout,
    };
    if let Err(control_error) = handle_generate_command(
        &mut worker_loop_state,
        active_generation_permit,
        generation_command,
        start_sender,
        stream_event_sender,
        &runtime.model_policy_catalog,
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

/// Handles an image generation command. Mirrors the inlined `GenerateImage`
/// arm, including the admission metadata (`admitted_at`, `queue_wait_elapsed`)
/// the image handler records.
async fn admit_image_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateImage {
        active_generation_permit,
        generation_command,
        start_sender,
        image_result_sender,
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
    if let Err(control_error) = handle_generate_image_command(
        &mut runtime.worker_process,
        active_generation_permit,
        generation_command,
        start_sender,
        image_result_sender,
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
        runtime.image_generation_timeouts,
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

/// Handles the shutdown command: shuts the worker down and reports the
/// outcome, then leaves the loop.
async fn handle_shutdown_command(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) -> WorkerLoopFlow {
    let WorkerLoopCommand::Shutdown { shutdown_sender } = routed_command else {
        return WorkerLoopFlow::Continue;
    };
    let shutdown_outcome = shutdown_worker(
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
    )
    .await;
    let _send_outcome = shutdown_sender.send(shutdown_outcome);
    WorkerLoopFlow::Exit
}

/// The shared guard for both worker-replacement commands: a busy active
/// generation rejects with `GenerationBusy`; an idle-but-unready worker
/// rejects with `MissingActiveWorker`; otherwise the replacement proceeds.
/// The two replacement arms used to repeat these two `if` blocks verbatim.
pub(in crate::worker) fn reject_worker_replacement_request(
    active_generation_is_busy: bool,
    is_ready: bool,
) -> Option<WorkerControlError> {
    if active_generation_is_busy {
        Some(WorkerControlError::GenerationBusy)
    } else if !is_ready {
        Some(WorkerControlError::MissingActiveWorker)
    } else {
        None
    }
}

/// Runs a `WorkerReplacement` and reports its result. Mirrors the inlined
/// `RestartWorker` arm.
async fn execute_worker_replacement(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::RestartWorker {
        worker_executable_path,
        model_policy_catalog: replacement_model_policy_catalog,
        worker_startup_configuration,
        expected_configuration_generation,
        restart_sender,
    } = routed_command
    else {
        return;
    };
    if let Some(control_error) =
        reject_worker_replacement_request(runtime.active_generation.is_some(), runtime.is_ready)
    {
        let _send_outcome = restart_sender.send(Err(control_error));
        return;
    }
    let replacement_result = WorkerReplacement::new(
        worker_executable_path,
        replacement_model_policy_catalog,
        worker_startup_configuration,
        expected_configuration_generation,
        runtime.model_load_timeout,
    )
    .execute(
        &mut runtime.worker_process,
        &mut runtime.model_policy_catalog,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
    )
    .await;
    let _send_outcome = restart_sender.send(replacement_result);
}

/// Queues the MLX memory-limit update while a generation is active, or applies
/// it immediately when idle. Mirrors the inlined `UpdateMlxMemoryLimit` arm,
/// including the pending-ceiling health publish on the queued path.
async fn queue_or_apply_mlx_memory_limit_update(
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
    let _send_outcome = update_sender.send(update_outcome);
}

/// Runs a memory-demotion restart and reports its result. Mirrors the inlined
/// `RestartWorkerForMemoryDemotion` arm; the busy/unready guard is the same
/// `reject_worker_replacement_request` the plain restart arm uses.
async fn execute_memory_demotion_restart(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) {
    // Stop-first replacement is invalid while a generation
    // is active; the memory endpoint holds admission and
    // waits for idle before sending this command.
    let WorkerLoopCommand::RestartWorkerForMemoryDemotion {
        worker_executable_path,
        model_policy_catalog: replacement_model_policy_catalog,
        worker_startup_configuration,
        expected_configuration_generation,
        demotion_sender,
    } = routed_command
    else {
        return;
    };
    if let Some(control_error) =
        reject_worker_replacement_request(runtime.active_generation.is_some(), runtime.is_ready)
    {
        let _send_outcome = demotion_sender.send(Err(control_error));
        return;
    }
    let demotion_result = MemoryDemotionRestart::new(
        worker_executable_path,
        replacement_model_policy_catalog,
        worker_startup_configuration,
        expected_configuration_generation,
        runtime.model_load_timeout,
    )
    .execute(
        &mut runtime.worker_process,
        &mut runtime.model_policy_catalog,
        &runtime.health_snapshot,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.active_generation,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
    )
    .await;
    let _send_outcome = demotion_sender.send(demotion_result);
}

/// Replaces the active model-policy catalog and reports success. Mirrors the
/// inlined `UpdateModelPolicyCatalog` arm.
fn replace_model_policy_catalog(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::UpdateModelPolicyCatalog {
        model_policy_catalog: replacement_model_policy_catalog,
        update_sender,
    } = routed_command
    else {
        return;
    };
    runtime.model_policy_catalog = replacement_model_policy_catalog;
    let _send_outcome = update_sender.send(Ok(()));
}

/// Delegates a prompt-cache clear to its handler with the loop's shared
/// mutation surface. Mirrors the inlined `ClearPromptCache` arm.
async fn clear_prompt_cache(runtime: &mut WorkerLoopRuntime, routed_command: WorkerLoopCommand) {
    let WorkerLoopCommand::ClearPromptCache {
        model_id,
        clear_sender,
    } = routed_command
    else {
        return;
    };
    handle_prompt_cache_clear_command(
        model_id,
        clear_sender,
        &mut runtime.pending_prompt_cache_clear,
        &mut runtime.worker_process,
        &runtime.health_snapshot,
        &mut runtime.active_generation,
        &mut runtime.is_ready,
        &mut runtime.model_load_deadline,
        &mut runtime.performance_log,
        &mut runtime.completion_log,
        &runtime.active_generation_permits,
        &runtime.generation_queue_permits,
    )
    .await;
}
