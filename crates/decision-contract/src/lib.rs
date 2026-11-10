//! The contract every decision model shares.
//!
//! A decision model reads a state and a set of typed questions and returns a
//! probability for every allowed answer in one forward pass, generating no
//! text. Model families differ in their network and in how they lay a
//! question out as tokens, but they agree on everything else: what a question
//! is, which definitions are malformed, how logits become calibrated
//! probabilities, and what a typed answer contains. That shared part lives
//! here.
//!
//! The crate deliberately has no MLX, Metal or tokenizer-library dependency.
//! The daemon validates requests with it, the command-line tool prints
//! answers with it, and the converters read calibration with it; none of them
//! should build the graphics-processor world to do so. Rendering reaches a
//! tokenizer only through the [`DecisionTokenEncoder`] trait.
//!
//! # Model families
//!
//! Each family has its own rendering module, because the families lay out the
//! same question differently in ways that change answers (option order, the
//! wording of the yes/no options, JSON separators). A family module holds
//! only that family's rendering rules; nothing family-specific belongs in the
//! shared modules.
//!
//! * [`laya_rendering`] and [`laya_sequence`]: an encoder with one marker
//!   token per option. Pinned to the reference implementation by the golden
//!   vectors in `src/laya_golden_reference_vectors.json`.

#![forbid(unsafe_code)]

pub mod answer;
pub mod calibration;
pub mod criterion_text;
pub mod failure;
pub mod laya_rendering;
pub mod laya_sequence;
pub mod ordered_json;
pub mod outcome;
pub mod python_json_text;
pub mod question;
pub mod system_one_questions;
pub mod system_one_response;
pub mod token_encoder;

pub use answer::{
    AnswerError, ChoiceAnswer, DecisionAnswer, NoulAnswer, OptionProbability, ScoreAnswer,
    answer_confidence, answer_from_probabilities, distribution_confidence,
};
pub use calibration::{
    BucketTemperature, KindTemperatures, TemperatureTable, calibrated_probabilities,
    clamp_temperature, temperature_bucket_key,
};
pub use failure::DecisionFailure;
pub use ordered_json::{OrderedJsonMember, OrderedJsonValue};
pub use outcome::{QuestionInputCoverage, QuestionOutcome};
pub use python_json_text::{JsonKeyOrder, JsonTextStyle, python_float_repr, python_json_text};
pub use question::{
    ChoiceOption, DecisionQuestion, DecisionQuestionBody, DecisionQuestionKind, NoulLabels,
};
pub use system_one_questions::{
    MAXIMUM_OPTIONS_PER_QUESTION, MAXIMUM_QUESTIONS_PER_REQUEST, QuestionDefinitionError,
    parse_system_one_questions,
};
pub use system_one_response::{system_one_answer, system_one_response};
pub use token_encoder::{DecisionTokenEncoder, TokenEncodingError};

#[cfg(test)]
mod laya_golden_vector_tests;
#[cfg(test)]
mod test_support;
