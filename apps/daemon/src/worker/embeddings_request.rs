//! Embeddings as a request kind that runs as one forward pass: what its
//! admission checks and dispatches beyond the shared steps in
//! [`request_kind_admission`].
//!
//! [`request_kind_admission`]: crate::worker::request_kind_admission

use std::sync::{Arc, RwLock};

use ipc_protocol::{EmbeddingsCommand, RequestId};

use crate::worker::loop_types::{
    ActiveEmbeddingsGeneration, ActiveWorkerRequest, WorkerLoopRuntime,
};
use crate::worker::request_kind_admission::{AdmittedRequest, RequestKind};
use crate::{
    EmbeddingsExecutionError, EmbeddingsOutput, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
};

pub(in crate::worker) struct EmbeddingsRequestKind;

impl RequestKind for EmbeddingsRequestKind {
    type Work = EmbeddingsCommand;
    type ResultMessage = Result<EmbeddingsOutput, EmbeddingsExecutionError>;
    type RequestBounds = ();
    const COMMAND_NAME: &'static str = "GenerateEmbeddings";
    // A forward pass sends no progress before its answer, so health keeps
    // showing the worker as idle, as it did before this admission was shared.
    const STARTED_ACTIVITY: WorkerActivity = WorkerActivity::Idle;

    fn requested_model(embeddings_command: &EmbeddingsCommand) -> &str {
        &embeddings_command.model
    }

    fn request_id(embeddings_command: &EmbeddingsCommand) -> RequestId {
        embeddings_command.request_id
    }

    fn request_bounds(_embeddings_command: &EmbeddingsCommand, _runtime: &WorkerLoopRuntime) {}

    fn resident_model_accepts(
        health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
        embeddings_command: &EmbeddingsCommand,
    ) -> bool {
        embeddings_command_supported(health_snapshot, embeddings_command.dimensions)
    }

    async fn dispatch(
        worker_process: &mut WorkerProcess,
        embeddings_command: EmbeddingsCommand,
    ) -> Result<(), WorkerControlError> {
        worker_process
            .start_embeddings_generation(embeddings_command)
            .await
    }

    fn active_request(
        admitted_request: AdmittedRequest<Self::ResultMessage>,
        _request_bounds: (),
    ) -> ActiveWorkerRequest {
        let execution_deadline = admitted_request.single_shot_execution_deadline();
        ActiveWorkerRequest::Embeddings(ActiveEmbeddingsGeneration {
            _active_generation_permit: admitted_request.active_generation_permit,
            request_id: admitted_request.request_id,
            terminal_received_at: None,
            execution_deadline,
            embeddings_result_sender: admitted_request.result_sender,
            terminal_outcome: None,
        })
    }
}

/// Checks the resident model's advertised embedding surface: the vector
/// width must be positive, and explicit dimensions may only truncate.
fn embeddings_command_supported(
    health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
    requested_dimensions: Option<u32>,
) -> bool {
    health_snapshot.read().ok().is_some_and(|snapshot| {
        snapshot
            .ready_model_capabilities
            .as_ref()
            .and_then(|capabilities| capabilities.embeddings.as_ref())
            .is_some_and(|embedding_capabilities| {
                embedding_capabilities.vector_width > 0
                    && requested_dimensions
                        .is_none_or(|dimensions| dimensions <= embedding_capabilities.vector_width)
            })
    })
}

#[cfg(test)]
#[path = "embeddings_request_tests.rs"]
mod embeddings_request_tests;
