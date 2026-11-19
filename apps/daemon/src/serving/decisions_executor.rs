//! Shared decisions result types for the daemon HTTP and IPC boundaries.
//!
//! Live dispatch lives on `ImageGenerationExecutor`, as it does for
//! embeddings: `ApplicationState` holds one executor trait object for every
//! kind of request, and a second trait would force every test executor to
//! grow another impl.

use decision_contract::decision_training_job::{DecisionTrainingOutcome, DecisionTrainingStage};
use decision_contract::{DecisionFailure, QuestionOutcome};
use ipc_protocol::{DecisionTrainingCommand, DecisionsCommand, RequestId, WorkerCommand};
use tokio::sync::mpsc;

/// What a decisions request delivers: its answers, or for a training,
/// progress messages and then the trained head.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionsOutput {
    /// Every answer of one decision request.
    Answered {
        /// One outcome per question, in request order.
        question_outcomes: Vec<QuestionOutcome>,
        /// The tokens the model read across all questions.
        input_token_count: u32,
        elapsed_millis: u64,
    },
    /// How far a training has got; it continues.
    TrainingProgress {
        stage: DecisionTrainingStage,
        completed: u64,
        total: u64,
    },
    /// A finished training.
    ModelTrained {
        outcome: Box<DecisionTrainingOutcome>,
        elapsed_millis: u64,
    },
}

/// The work one decisions request asks of the worker.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionsWork {
    Answer(DecisionsCommand),
    TrainHead(DecisionTrainingCommand),
}

impl DecisionsWork {
    #[must_use]
    pub fn request_id(&self) -> RequestId {
        match self {
            Self::Answer(decisions_command) => decisions_command.request_id,
            Self::TrainHead(training_command) => training_command.request_id,
        }
    }

    #[must_use]
    pub fn model(&self) -> &str {
        match self {
            Self::Answer(decisions_command) => &decisions_command.model,
            Self::TrainHead(training_command) => &training_command.model,
        }
    }

    #[must_use]
    pub fn into_worker_command(self) -> WorkerCommand {
        match self {
            Self::Answer(decisions_command) => WorkerCommand::GenerateDecisions(decisions_command),
            Self::TrainHead(training_command) => {
                WorkerCommand::TrainDecisionModel(training_command)
            }
        }
    }
}

/// Failure delivered after a decisions request was admitted to the worker.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionsExecutionError {
    WorkerFailure(DecisionFailure),
    WorkerUnavailable,
}

/// Messages one decisions request's channel holds: a training's progress
/// ahead of its result.
pub(crate) const DECISIONS_RESULT_CHANNEL_CAPACITY: usize = 8;

/// The channel one decisions request's progress and terminal outcome are delivered on.
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
