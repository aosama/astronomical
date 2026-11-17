//! The stub daemon's answers to decision-model requests: the scripted
//! outcome of a decide request, and its refusal to train, since the stub
//! serves no model.

use astronomical_decision_contract::DecisionFailure;
use astronomical_ipc_protocol::DaemonResponse;

use super::stub_daemon::StubDecisionsOutcome;

/// The one terminal frame the scripted outcome stands for.
pub fn scripted_decisions_response(
    decisions_outcome: &StubDecisionsOutcome,
    answering_model_id: String,
) -> DaemonResponse {
    match decisions_outcome {
        StubDecisionsOutcome::Completed(question_outcomes) => DaemonResponse::DecisionsCompleted {
            model: answering_model_id,
            question_outcomes: question_outcomes.clone(),
            input_token_count: 42,
            elapsed_millis: 7,
        },
        StubDecisionsOutcome::Failed(failure_reason) => DaemonResponse::DecisionsFailed {
            reason: failure_reason.clone(),
        },
        StubDecisionsOutcome::Rejected(reason) => DaemonResponse::GenerationRejected {
            reason: reason.clone(),
        },
    }
}

/// The stub serves no model, so a training is always refused.
pub fn training_refusal() -> DaemonResponse {
    DaemonResponse::DecisionsFailed {
        reason: DecisionFailure::invalid_request("the stub daemon does not train"),
    }
}
