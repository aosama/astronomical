//! Decisions as a single-shot request kind: what its admission checks and
//! dispatches beyond the shared steps in [`single_shot_request`].
//!
//! [`single_shot_request`]: crate::worker::single_shot_request

use std::sync::{Arc, RwLock};

use ipc_protocol::RequestId;

use crate::serving::decisions_executor::DecisionsWork;
use crate::worker::loop_types::{ActiveDecisionsGeneration, ActiveWorkerRequest};
use crate::worker::single_shot_request::{AdmittedSingleShot, SingleShotRequestKind};
use crate::{
    DecisionsExecutionError, DecisionsOutput, WorkerControlError, WorkerHealthSnapshot,
    WorkerProcess,
};

pub(in crate::worker) struct DecisionsRequestKind;

impl SingleShotRequestKind for DecisionsRequestKind {
    type Work = DecisionsWork;
    type ResultMessage = Result<DecisionsOutput, DecisionsExecutionError>;
    const COMMAND_NAME: &'static str = "GenerateDecisions";

    fn requested_model(decisions_work: &DecisionsWork) -> &str {
        decisions_work.model()
    }

    fn request_id(decisions_work: &DecisionsWork) -> RequestId {
        decisions_work.request_id()
    }

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
        admitted_request: AdmittedSingleShot<Self::ResultMessage>,
    ) -> ActiveWorkerRequest {
        ActiveWorkerRequest::Decisions(ActiveDecisionsGeneration {
            _active_generation_permit: admitted_request.active_generation_permit,
            request_id: admitted_request.request_id,
            terminal_received_at: None,
            execution_deadline: admitted_request.execution_deadline,
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
