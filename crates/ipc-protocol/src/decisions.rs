//! Typed decision requests and worker-boundary validation.
//!
//! A decision model answers typed questions about a state in one forward
//! pass, so the command carries the state and the questions and the worker
//! returns one outcome per question. Both travel as ordered JSON: the order
//! of object members is part of the text the model reads, and it must
//! survive both process boundaries exactly as the caller wrote it.

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, QuestionDefinitionError, parse_system_one_questions,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::RequestId;

/// One decision request sent to the local inference worker.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionsCommand {
    pub request_id: RequestId,
    pub model: String,
    /// What to decide about: text, or any JSON value the model reads as JSON text.
    pub state: OrderedJsonValue,
    /// The System One `questions` object, exactly as the caller wrote it.
    pub questions: OrderedJsonValue,
}

impl DecisionsCommand {
    /// Independently validates a decision request after it crosses the worker
    /// trust boundary, returning its questions parsed.
    pub fn validated_questions(&self) -> Result<Vec<DecisionQuestion>, DecisionsValidationError> {
        if self.model.trim().is_empty() {
            return Err(DecisionsValidationError::EmptyModelId);
        }
        Ok(parse_system_one_questions(&self.questions)?)
    }
}

/// Rejection reasons enforced independently at the worker IPC boundary.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum DecisionsValidationError {
    #[error("model id must not be empty")]
    EmptyModelId,
    #[error(transparent)]
    Questions(#[from] QuestionDefinitionError),
}

/// What a loaded decision model can be asked.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerDecisionCapabilities {
    /// The longest sequence one question may occupy: its instructions, its
    /// options and as much of the state as still fits.
    pub maximum_tokens_per_question: u32,
}

/// A sanity floor for the per-question token limit a model advertises. No
/// question fits below it once its separators, type marker and two option
/// markers are counted, so a smaller limit means the advertisement is wrong
/// and the model is treated as unable to decide.
pub const MINIMUM_DECISION_TOKENS_PER_QUESTION: u32 = 8;

impl WorkerDecisionCapabilities {
    /// Whether the advertised limit can hold a question.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        self.maximum_tokens_per_question >= MINIMUM_DECISION_TOKENS_PER_QUESTION
    }
}
