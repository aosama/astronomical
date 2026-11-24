//! Typed answers built from a question's calibrated option probabilities.
//!
//! Every model family ends with the same thing: one probability per allowed
//! answer. Turning those into a chosen option, an expected score or a
//! probability of "true", and into the two confidence numbers callers gate
//! on, is identical across families, so it is done once here.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::criterion_text::criterion_text;
use crate::ordered_json::OrderedJsonValue;
use crate::question::{
    ChoiceOption, DecisionQuestion, DecisionQuestionBody, NOUL_FALSE_INDEX, NOUL_TRUE_INDEX,
};

/// The answer to one question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "answer", rename_all = "snake_case")]
pub enum DecisionAnswer {
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Noul(NoulAnswer),
}

/// The answer to a `choice` question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    /// The key of the most probable option; the first one when several tie.
    pub chosen_key: String,
    /// Every option's probability, in the order the question listed them.
    pub option_probabilities: Vec<OptionProbability>,
    /// How concentrated the whole distribution is; see [`distribution_confidence`].
    pub distribution_confidence: f64,
    /// The probability of the chosen option; see [`answer_confidence`].
    pub answer_confidence: f64,
}

/// One option's key and probability.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OptionProbability {
    pub key: String,
    pub probability: f64,
}

/// The answer to a `score` question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreAnswer {
    /// The probability-weighted average of the level indices, so it can fall between levels.
    pub expected_level: f64,
    /// The text of each level, index 0 first, as the model was shown it.
    pub level_legend: Vec<String>,
    /// Every level's probability, index 0 first.
    pub level_probabilities: Vec<f64>,
    pub distribution_confidence: f64,
    pub answer_confidence: f64,
}

/// The answer to a `noul` question.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulAnswer {
    /// The probability that the statement holds.
    pub probability_true: f64,
    /// The probability of the more likely side: `max(p, 1 - p)`.
    pub answer_confidence: f64,
}

/// Why a probability list cannot answer a question.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AnswerError {
    #[error(
        "question {question:?} allows {expected_option_count} answers but {actual_probability_count} probabilities were produced"
    )]
    ProbabilityCountMismatch {
        question: String,
        expected_option_count: usize,
        actual_probability_count: usize,
    },
    #[error("question {question:?} defines no options, so it has no answer")]
    NoOptions { question: String },
}

/// The probability of the answer being reported: the largest probability.
///
/// This is the number temperature calibration fits. After calibration on
/// representative data, about `c` of the answers reported at confidence `c`
/// are right, which makes it the number to compare against a threshold.
#[must_use]
pub fn answer_confidence(option_probabilities: &[f64]) -> f64 {
    if option_probabilities.is_empty() {
        return 1.0;
    }
    let mut largest_probability = 0.0_f64;
    for option_probability in option_probabilities {
        largest_probability = largest_probability.max(*option_probability);
    }
    largest_probability.clamp(0.0, 1.0)
}

/// The smallest probability the entropy sum takes a logarithm of.
///
/// A probability that underflowed to zero would otherwise contribute
/// `0 * ln(0)`, which is `NaN`.
const ENTROPY_PROBABILITY_FLOOR: f64 = 1e-12;

/// How concentrated a distribution is: one minus its normalized Shannon entropy.
///
/// 1 means all the probability is on one option and 0 means it is spread
/// evenly. It describes the whole distribution and is not calibrated, so it
/// must not be compared against the same threshold as [`answer_confidence`].
/// Fewer than two options have no entropy scale, and report 1.
#[must_use]
pub fn distribution_confidence(option_probabilities: &[f64]) -> f64 {
    let option_count = option_probabilities.len();
    if option_count < 2 {
        return 1.0;
    }
    let mut entropy = 0.0_f64;
    for option_probability in option_probabilities {
        let floored_probability = option_probability.clamp(ENTROPY_PROBABILITY_FLOOR, 1.0);
        entropy -= option_probability * floored_probability.ln();
    }
    let maximum_entropy = (option_count as f64).ln();
    (1.0 - entropy / maximum_entropy).clamp(0.0, 1.0)
}

/// Builds the typed answer to `question` from its option probabilities.
///
/// `option_probabilities` is in canonical order: a `choice` in the order the
/// question listed its options, a `score` from level 0 upward, and a `noul`
/// as `[false, true]`. A model family that renders options in another order
/// reorders its probabilities before calling this.
pub fn answer_from_probabilities(
    question: &DecisionQuestion,
    option_probabilities: &[f64],
) -> Result<DecisionAnswer, AnswerError> {
    let expected_option_count = question.option_count();
    if option_probabilities.len() != expected_option_count {
        return Err(AnswerError::ProbabilityCountMismatch {
            question: question.name.clone(),
            expected_option_count,
            actual_probability_count: option_probabilities.len(),
        });
    }
    if expected_option_count == 0 {
        return Err(AnswerError::NoOptions {
            question: question.name.clone(),
        });
    }
    Ok(match &question.body {
        DecisionQuestionBody::Choice { options } => {
            DecisionAnswer::Choice(choice_answer(options, option_probabilities))
        }
        DecisionQuestionBody::Score { levels } => {
            DecisionAnswer::Score(score_answer(levels, option_probabilities))
        }
        DecisionQuestionBody::Noul { .. } => {
            DecisionAnswer::Noul(noul_answer(option_probabilities))
        }
    })
}

fn choice_answer(options: &[ChoiceOption], option_probabilities: &[f64]) -> ChoiceAnswer {
    let mut keyed_probabilities = Vec::with_capacity(options.len());
    for (option, option_probability) in options.iter().zip(option_probabilities) {
        keyed_probabilities.push(OptionProbability {
            key: option.key.clone(),
            probability: *option_probability,
        });
    }
    let chosen_index = first_most_probable_index(option_probabilities);
    ChoiceAnswer {
        chosen_key: options[chosen_index].key.clone(),
        option_probabilities: keyed_probabilities,
        distribution_confidence: distribution_confidence(option_probabilities),
        answer_confidence: answer_confidence(option_probabilities),
    }
}

fn score_answer(levels: &[OrderedJsonValue], level_probabilities: &[f64]) -> ScoreAnswer {
    let mut level_legend = Vec::with_capacity(levels.len());
    for level in levels {
        level_legend.push(criterion_text(level));
    }
    ScoreAnswer {
        expected_level: expected_level(level_probabilities),
        level_legend,
        level_probabilities: level_probabilities.to_vec(),
        distribution_confidence: distribution_confidence(level_probabilities),
        answer_confidence: answer_confidence(level_probabilities),
    }
}

/// `canonical_probabilities` is `[false, true]`; the caller has checked its length.
fn noul_answer(canonical_probabilities: &[f64]) -> NoulAnswer {
    let probability_true = canonical_probabilities[NOUL_TRUE_INDEX];
    let probability_false = canonical_probabilities[NOUL_FALSE_INDEX];
    NoulAnswer {
        probability_true,
        answer_confidence: probability_true.max(probability_false).clamp(0.0, 1.0),
    }
}

/// The index of the largest probability; the earliest one when several are equal.
fn first_most_probable_index(option_probabilities: &[f64]) -> usize {
    let mut most_probable_index = 0;
    for (option_index, option_probability) in option_probabilities.iter().enumerate() {
        if *option_probability > option_probabilities[most_probable_index] {
            most_probable_index = option_index;
        }
    }
    most_probable_index
}

fn expected_level(level_probabilities: &[f64]) -> f64 {
    let mut probability_weighted_level = 0.0_f64;
    for (level_index, level_probability) in level_probabilities.iter().enumerate() {
        probability_weighted_level += level_index as f64 * level_probability;
    }
    probability_weighted_level
}

#[cfg(test)]
#[path = "answer_tests.rs"]
mod tests;
