//! Worker-loop command arms for the requests that run as one forward pass:
//! decisions and embeddings.
//!
//! Each arm unpacks its command and hands it to the shared single-shot
//! admission, which refuses while the worker is unavailable, lets a pending
//! memory-limit change settle, and contains the worker when admission fails.
//! They live apart from the chat and image arms, whose requests stream
//! progress.

use crate::worker::decisions_request::DecisionsRequestKind;
use crate::worker::embeddings_request::EmbeddingsRequestKind;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::worker::single_shot_request::{SingleShotCommand, admit_single_shot_command};

/// Admits a request that runs as one forward pass: a decisions command
/// through its own arm, and an embeddings command through the other. Any
/// other command is ignored there, as each arm ignores what it does not own.
pub(in crate::worker) async fn admit_single_shot_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    if matches!(routed_command, WorkerLoopCommand::GenerateDecisions { .. }) {
        admit_decisions_command(runtime, is_worker_running, routed_command).await;
        return;
    }
    admit_embeddings_command(runtime, is_worker_running, routed_command).await;
}

async fn admit_decisions_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateDecisions {
        active_generation_permit,
        decisions_work,
        start_sender,
        decisions_result_sender,
    } = routed_command
    else {
        return;
    };
    let decisions_command = SingleShotCommand::<DecisionsRequestKind> {
        active_generation_permit,
        work: decisions_work,
        start_sender,
        result_sender: decisions_result_sender,
    };
    admit_single_shot_command(runtime, is_worker_running, decisions_command).await;
}

async fn admit_embeddings_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateEmbeddings {
        active_generation_permit,
        embeddings_command,
        start_sender,
        embeddings_result_sender,
    } = routed_command
    else {
        return;
    };
    let embeddings_command = SingleShotCommand::<EmbeddingsRequestKind> {
        active_generation_permit,
        work: embeddings_command,
        start_sender,
        result_sender: embeddings_result_sender,
    };
    admit_single_shot_command(runtime, is_worker_running, embeddings_command).await;
}
