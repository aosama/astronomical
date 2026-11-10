//! How Kev turns option probabilities into answers: the shared answer, with
//! Kev's own confidences and score legend.
//!
//! The confidences mirror TypeSafe's reference adapter, as `kev/api.py`'s
//! `choice_confidence` and `score_confidence` do: a choice's is how far its
//! top probability sits above uniform, a score's how tightly the probability
//! gathers around its most likely level.

use crate::answer::{AnswerError, DecisionAnswer, answer_from_probabilities};
use crate::kev_rendering::kev_question_text;
use crate::question::DecisionQuestion;

/// The answer Kev gives for `option_probabilities`: the shared answer, with
/// Kev's own confidences and a score legend in Kev's own text.
pub fn kev_answer_from_probabilities(
    question: &DecisionQuestion,
    option_probabilities: &[f64],
) -> Result<DecisionAnswer, AnswerError> {
    let mut answer = answer_from_probabilities(question, option_probabilities)?;
    match &mut answer {
        DecisionAnswer::Choice(choice_answer) => {
            choice_answer.distribution_confidence = kev_choice_confidence(option_probabilities);
        }
        DecisionAnswer::Score(score_answer) => {
            score_answer.distribution_confidence = kev_score_confidence(option_probabilities);
            score_answer.level_legend = kev_question_text(question).option_texts;
        }
        DecisionAnswer::Noul(_) => {}
    }
    Ok(answer)
}

/// `_normalize`: the probabilities scaled to sum to one, or uniform when
/// they sum to zero.
fn normalized(option_probabilities: &[f64]) -> Vec<f64> {
    let probability_sum: f64 = option_probabilities.iter().sum();
    let option_count = option_probabilities.len() as f64;
    if probability_sum == 0.0 {
        return vec![1.0 / option_count; option_probabilities.len()];
    }
    option_probabilities
        .iter()
        .map(|probability| probability / probability_sum)
        .collect()
}

/// `choice_confidence`: `(p_max − 1/K) / (1 − 1/K)`, 0 at uniform and 1 at
/// certainty; one option is always certain.
#[must_use]
pub fn kev_choice_confidence(option_probabilities: &[f64]) -> f64 {
    let option_count = option_probabilities.len() as f64;
    if option_probabilities.len() <= 1 {
        return 1.0;
    }
    let largest_probability = normalized(option_probabilities)
        .into_iter()
        .fold(0.0, f64::max);
    (largest_probability - 1.0 / option_count) / (1.0 - 1.0 / option_count)
}

/// `score_confidence`: `max(0, 1 − E|level − mode| / D)`, where `mode` is
/// the first most likely level and `D` the mean distance of a uniform
/// distribution from its centre.
#[must_use]
pub fn kev_score_confidence(level_probabilities: &[f64]) -> f64 {
    let level_count = level_probabilities.len();
    if level_count <= 1 {
        return 1.0;
    }
    let normalized_probabilities = normalized(level_probabilities);
    let mut mode_level = 0;
    for (level_index, level_probability) in normalized_probabilities.iter().enumerate() {
        if *level_probability > normalized_probabilities[mode_level] {
            mode_level = level_index;
        }
    }
    let centre = (level_count - 1) as f64 / 2.0;
    let mut uniform_spread = 0.0;
    let mut expected_distance = 0.0;
    for (level_index, level_probability) in normalized_probabilities.iter().enumerate() {
        uniform_spread += (level_index as f64 - centre).abs();
        expected_distance += level_probability * (level_index as f64 - mode_level as f64).abs();
    }
    uniform_spread /= level_count as f64;
    (1.0 - expected_distance / uniform_spread).max(0.0)
}

#[cfg(test)]
#[path = "kev_answers_tests.rs"]
mod tests;
