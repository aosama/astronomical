//! Decisions as a request kind that runs as one forward pass: what its
//! admission checks and dispatches beyond the shared steps in
//! [`request_kind_admission`].
//!
//! [`request_kind_admission`]: crate::worker::request_kind_admission

use std::sync::{Arc, RwLock};

use ipc_protocol::RequestId;

use crate::serving::decisions_executor::DecisionsWork;
use crate::worker::loop_types::WorkerLoopRuntime;
use crate::worker::loop_types::{ActiveDecisionsGeneration, ActiveWorkerRequest};
use crate::worker::request_kind_admission::{AdmittedRequest, RequestKind};
use crate::{
    DecisionsExecutionError, DecisionsOutput, WorkerActivity, WorkerControlError,
    WorkerHealthSnapshot, WorkerProcess,
};

pub(in crate::worker) struct DecisionsRequestKind;

impl RequestKind for DecisionsRequestKind {
    type Work = DecisionsWork;
    type ResultMessage = Result<DecisionsOutput, DecisionsExecutionError>;
    type RequestBounds = ();
    const COMMAND_NAME: &'static str = "GenerateDecisions";
    const STARTED_ACTIVITY: WorkerActivity = WorkerActivity::Idle;

    fn requested_model(decisions_work: &DecisionsWork) -> &str {
        decisions_work.model()
    }

    fn request_id(decisions_work: &DecisionsWork) -> RequestId {
        decisions_work.request_id()
    }

    fn request_bounds(_decisions_work: &DecisionsWork, _runtime: &WorkerLoopRuntime) {}

    fn resident_model_accepts(
        health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>,
        _decisions_work: &DecisionsWork,
    ) -> bool {
        resident_model_answers_decisions(health_snapshot)
    }

    async fn dispatch(
        worker_process: &mut WorkerProcess,
        decisions_work: DecisionsWork,
    ) -> Result<(), WorkerControlError> {
        worker_process.start_decisions_work(decisions_work).await
    }

    fn active_request(
        admitted_request: AdmittedRequest<Self::ResultMessage>,
        _request_bounds: (),
    ) -> ActiveWorkerRequest {
        let execution_deadline = admitted_request.single_shot_execution_deadline();
        ActiveWorkerRequest::Decisions(ActiveDecisionsGeneration {
            _active_generation_permit: admitted_request.active_generation_permit,
            request_id: admitted_request.request_id,
            terminal_received_at: None,
            execution_deadline,
            decisions_result_sender: admitted_request.result_sender,
            terminal_outcome: None,
        })
    }
}

/// Whether the resident model advertises a usable decisions surface.
fn resident_model_answers_decisions(health_snapshot: &Arc<RwLock<WorkerHealthSnapshot>>) -> bool {
    let Ok(snapshot) = health_snapshot.read() else {
        return false;
    };
    let Some(capabilities) = snapshot.ready_model_capabilities.as_ref() else {
        return false;
    };
    match capabilities.decisions {
        Some(decision_capabilities) => decision_capabilities.is_usable(),
        None => false,
    }
}

#[cfg(test)]
#[path = "decisions_request_tests.rs"]
mod decisions_request_tests;
