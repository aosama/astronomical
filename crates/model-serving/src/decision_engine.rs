//! Architecture-neutral lifecycle contract implemented by a concrete decision runtime.
//!
//! A decision model answers typed questions about a state in one forward
//! pass, so like the embedding runtime there is no step loop and no
//! mid-request cancellation: one call returns every answer or one failure.

use crate::memory::telemetry::MlxMemoryTelemetry;
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, OrderedJsonValue, QuestionOutcome,
};

/// A loaded decision runtime that answers typed questions about a state.
pub trait DecisionEngine: 'static {
    /// Loads model resources and returns the identity and limits to advertise.
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure>;

    /// Answers every question about `state`, in the order the questions are given.
    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEngineOutput, DecisionFailure>;

    /// The smallest MLX memory ceiling this model can be served under.
    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        1
    }

    /// Returns the final MLX observation captured after request cleanup, when available.
    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        None
    }

    /// Collects an idle memory observation without adding decision methods.
    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        None
    }
}

/// Loaded decision-model identity and its per-question token limits.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionEngineLoadResult {
    pub model_id: String,
    /// The longest sequence one question may occupy: its instructions, its
    /// options and as much of the state as still fits.
    pub maximum_tokens_per_question: u32,
}

/// Every answer of one decision request.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionEngineOutput {
    /// One entry per question, in request order.
    pub question_outcomes: Vec<QuestionOutcome>,
    /// The tokens the model read across all questions.
    pub input_token_count: u32,
    pub elapsed_millis: u64,
}

/// Default decision slot used by workers whose models answer no decision requests.
#[doc(hidden)]
pub struct DecisionUnavailableEngine;

impl DecisionEngine for DecisionUnavailableEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        Err(DecisionFailure::fatal_execution(
            "the selected model does not answer decision requests",
        ))
    }

    fn decide(
        &mut self,
        _state: &OrderedJsonValue,
        _questions: &[DecisionQuestion],
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        Err(DecisionFailure::fatal_execution(
            "the selected model does not answer decision requests",
        ))
    }
}

/// A boxed engine is an engine, so a worker can hold whichever decision
/// family its factory built without naming each family in its type.
impl DecisionEngine for Box<dyn DecisionEngine> {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        self.as_mut().load()
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        self.as_mut().decide(state, questions)
    }

    fn minimum_mlx_memory_ceiling_bytes(&self) -> u64 {
        self.as_ref().minimum_mlx_memory_ceiling_bytes()
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.as_mut().take_post_cleanup_memory_telemetry()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        self.as_ref().collect_mlx_memory_telemetry()
    }
}
