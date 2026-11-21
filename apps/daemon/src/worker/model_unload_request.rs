//! Unloading the loaded model on request (`POST /v1/control/unload-model`).
//!
//! It frees the model's memory the same way the idle unload does, by
//! replacing the worker process with an empty one: MLX does not hand wired
//! memory back reliably while its process lives. The next request loads its
//! model into the fresh worker as usual.
//!
//! A request in progress or waiting for admission is never cut short: the
//! unload answers `Busy` and leaves the model loaded, so a person's reply is
//! not lost to a tool that asked for memory back.

use tokio::sync::oneshot;

use crate::worker::idle_model_unload::{ModelUnloadCause, WorkerIdleness, unload_loaded_model};
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};

/// What an unload request did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelUnloadOutcome {
    /// The model was unloaded and an empty worker is ready.
    Unloaded { model_id: String },
    /// No model was loaded, so nothing changed.
    NothingLoaded,
    /// A request is being served or is waiting, or a model is loading, so
    /// the model stays loaded.
    Busy,
    /// The worker is not running (it stopped or is being relaunched after a
    /// failure), so there is nothing to unload from.
    Unavailable,
    /// The model was unloaded, but the empty worker did not start; the
    /// worker's own relaunch retries it.
    UnloadedWithoutReadyWorker { model_id: String },
}

/// Whether the worker process is running, and whether it is ready to serve
/// (a running worker that is not ready is loading a model).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WorkerState {
    pub(crate) is_running: bool,
    pub(crate) is_ready: bool,
}

/// What the loop should do with an unload request, decided from its state
/// alone so the rule is tested without a worker process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ModelUnloadDecision {
    Unload { model_id: String },
    Answer(ModelUnloadOutcome),
}

/// Unload only a loaded model, only when no request holds or waits for the
/// worker. A worker still loading a model counts as busy: its request is on
/// its way.
pub(crate) fn decide_model_unload(
    ready_model_id: Option<String>,
    worker_state: WorkerState,
    worker_idleness: &WorkerIdleness,
) -> ModelUnloadDecision {
    if !worker_state.is_running {
        return ModelUnloadDecision::Answer(ModelUnloadOutcome::Unavailable);
    }
    if !worker_state.is_ready {
        return ModelUnloadDecision::Answer(ModelUnloadOutcome::Busy);
    }
    let Some(model_id) = ready_model_id else {
        return ModelUnloadDecision::Answer(ModelUnloadOutcome::NothingLoaded);
    };
    if worker_idleness.is_free_to_unload() {
        ModelUnloadDecision::Unload { model_id }
    } else {
        ModelUnloadDecision::Answer(ModelUnloadOutcome::Busy)
    }
}

/// Runs one `UnloadModel` command inside the worker loop.
pub(in crate::worker) async fn handle_model_unload_command(
    runtime: &mut WorkerLoopRuntime,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::UnloadModel { unload_sender } = routed_command else {
        return;
    };
    let ready_model_id = runtime
        .health_snapshot
        .read()
        .ok()
        .and_then(|worker_health_snapshot| worker_health_snapshot.ready_model_id.clone());
    let worker_idleness = WorkerIdleness {
        has_loaded_model: ready_model_id.is_some(),
        has_active_request: runtime.active_generation.is_some(),
        free_active_generation_permits: runtime.active_generation_permits.available_permits(),
        free_generation_queue_permits: runtime.generation_queue_permits.available_permits(),
    };
    let worker_state = WorkerState {
        is_running: runtime.worker_process.process_id().is_some(),
        is_ready: runtime.is_ready,
    };
    let outcome = match decide_model_unload(ready_model_id, worker_state, &worker_idleness) {
        ModelUnloadDecision::Answer(outcome) => outcome,
        ModelUnloadDecision::Unload { model_id } => {
            if unload_loaded_model(runtime, ModelUnloadCause::Requested).await {
                ModelUnloadOutcome::Unloaded { model_id }
            } else {
                ModelUnloadOutcome::UnloadedWithoutReadyWorker { model_id }
            }
        }
    };
    send_unload_outcome(unload_sender, outcome);
}

fn send_unload_outcome(
    unload_sender: oneshot::Sender<ModelUnloadOutcome>,
    outcome: ModelUnloadOutcome,
) {
    // A caller that stopped waiting no longer needs the answer; the unload
    // has happened either way.
    let _caller_may_have_gone = unload_sender.send(outcome);
}
