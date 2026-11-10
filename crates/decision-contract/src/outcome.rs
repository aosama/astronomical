//! One question's answer together with what the model read to produce it.
//!
//! These types travel unchanged from a decision engine, across the worker
//! and daemon process boundaries, to the REST and command-line surfaces, so
//! they live here with the rest of the shared contract.

use serde::{Deserialize, Serialize};

use crate::answer::DecisionAnswer;

/// One question's answer and what the model actually read to produce it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

impl QuestionInputCoverage {
    /// Whether the model read less than the whole state for this question.
    #[must_use]
    pub const fn state_was_cut(&self) -> bool {
        self.state_tokens_read < self.state_token_count
    }
}

#[cfg(test)]
#[path = "outcome_tests.rs"]
mod tests;
