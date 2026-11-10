//! Architecture-neutral lifecycle contract implemented by a concrete decision runtime.
//!
//! A decision model answers typed questions about a state in one forward
//! pass, so like the embedding runtime there is no step loop and no
//! mid-request cancellation: one call returns every answer or one failure.

use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, DecisionQuestion, OrderedJsonValue,
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

/// One question's answer and what the model actually read to produce it.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestionOutcome {
    pub question_name: String,
    pub answer: DecisionAnswer,
    /// The uncalibrated logit of each option, in canonical option order.
    pub option_logits: Vec<f32>,
    /// The temperature the logits were divided by before the softmax.
    pub applied_temperature: f64,
    pub input_coverage: QuestionInputCoverage,
}

/// How much of a question's input fit the model's token limits.
///
/// A model reads a bounded sequence, so a long state or many long options are
/// cut. The answer is still returned; these numbers let a caller see that it
/// was made from partial input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuestionInputCoverage {
    /// How many tokens the whole state encodes to.
    pub state_token_count: u32,
    /// How many of them the model read for this question.
    pub state_tokens_read: u32,
    /// The token cap applied to every option when the options overran their
    /// budget; `None` when no option was cut for that reason.
    pub option_token_cap: Option<u32>,
    /// How many options still read differently from every other option.
    /// Fewer than the option count means the cut made options indistinguishable.
    pub distinguishable_option_count: u32,
}
