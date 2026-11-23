//! Worker-loop command arms for the request kinds that share one admission:
//! decisions, embeddings and speech.
//!
//! Each arm unpacks its command and hands it to
//! [`admit_request_kind_command`], which refuses while the worker is
//! unavailable or busy, lets a pending memory-limit change settle, admits the
//! request, and contains the worker when admission fails. They live apart
//! from the chat and image arms, which keep their own admission.

use crate::worker::decisions_request::DecisionsRequestKind;
use crate::worker::embeddings_request::EmbeddingsRequestKind;
use crate::worker::loop_types::{WorkerLoopCommand, WorkerLoopRuntime};
use crate::worker::request_kind_admission::{RequestKindCommand, admit_request_kind_command};
use crate::worker::speech_request::SpeechRequestKind;

/// Admits a decisions, embeddings or speech command through its own arm. Any
/// other command is ignored there, as each arm ignores what it does not own.
pub(in crate::worker) async fn admit_request_kind_generation_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    match routed_command {
        WorkerLoopCommand::GenerateDecisions { .. } => {
            admit_decisions_command(runtime, is_worker_running, routed_command).await;
        }
        WorkerLoopCommand::GenerateSpeech { .. } => {
            admit_speech_command(runtime, is_worker_running, routed_command).await;
        }
        _ => admit_embeddings_command(runtime, is_worker_running, routed_command).await,
    }
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
    let decisions_command = RequestKindCommand::<DecisionsRequestKind> {
        active_generation_permit,
        work: decisions_work,
        start_sender,
        result_sender: decisions_result_sender,
    };
    admit_request_kind_command(runtime, is_worker_running, decisions_command).await;
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
    let embeddings_command = RequestKindCommand::<EmbeddingsRequestKind> {
        active_generation_permit,
        work: embeddings_command,
        start_sender,
        result_sender: embeddings_result_sender,
    };
    admit_request_kind_command(runtime, is_worker_running, embeddings_command).await;
}

async fn admit_speech_command(
    runtime: &mut WorkerLoopRuntime,
    is_worker_running: bool,
    routed_command: WorkerLoopCommand,
) {
    let WorkerLoopCommand::GenerateSpeech {
        active_generation_permit,
        speech_command,
        start_sender,
        speech_result_sender,
    } = routed_command
    else {
        return;
    };
    let speech_command = RequestKindCommand::<SpeechRequestKind> {
        active_generation_permit,
        work: speech_command,
        start_sender,
        result_sender: speech_result_sender,
    };
    admit_request_kind_command(runtime, is_worker_running, speech_command).await;
}
