//! Shared decisions result types for the daemon HTTP and IPC boundaries.
//!
//! Live dispatch lives on `ImageGenerationExecutor`, as it does for
//! embeddings: `ApplicationState` holds one executor trait object for every
//! kind of request, and a second trait would force every test executor to
//! grow another impl.

use astronomical_decision_contract::{DecisionFailure, QuestionOutcome};
use tokio::sync::mpsc;

/// Every answer of one decision request.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionsOutput {
    /// One outcome per question, in request order.
    pub question_outcomes: Vec<QuestionOutcome>,
    /// The tokens the model read across all questions.
    pub input_token_count: u32,
    pub elapsed_millis: u64,
}

/// Failure delivered after a decisions request was admitted to the worker.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionsExecutionError {
    WorkerFailure(DecisionFailure),
    WorkerUnavailable,
}

/// The channel one decisions request's terminal outcome is delivered on.
pub(crate) type DecisionsResultSender =
    mpsc::Sender<Result<DecisionsOutput, DecisionsExecutionError>>;

/// Resolves when the caller disconnects from an in-flight decisions request.
pub(crate) async fn wait_for_decisions_disconnect(
    decisions_result_sender: Option<DecisionsResultSender>,
) {
    let Some(decisions_result_sender) = decisions_result_sender else {
        std::future::pending::<()>().await;
        return;
    };
    decisions_result_sender.closed().await;
}
