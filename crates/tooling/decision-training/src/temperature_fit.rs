//! Fitting one temperature to a set of scored questions.
//!
//! The fitted temperature is the one that minimises the mean negative
//! log-likelihood of the right answers. As a function of `1 / T` that loss
//! is convex (a log-sum-exp of linear terms minus a linear term), so it has
//! one minimum and a golden-section search finds it without gradients.
//!
//! The search runs over `ln T` rather than `T`, because the useful scale is
//! multiplicative: going from 0.5 to 1 matters as much as going from 2.5 to 5.
//! It is confined to the range engines apply, 0.5 to 5. A fit outside that
//! range would be clamped at serving time, so the search must optimise the
//! value that will actually be served, not one that will be cut down later.

use decision_contract::calibration::{MAXIMUM_APPLIED_TEMPERATURE, MINIMUM_APPLIED_TEMPERATURE};

use crate::scored_question::ScoredQuestion;
use crate::temperature_softmax::log_probabilities_at;

/// `(√5 − 1) / 2`: the fraction golden-section search keeps each step.
const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;
/// Each step shrinks the bracket to 0.618 of itself; 60 steps take the
/// `ln T` bracket of about 2.3 below 1e-12, far finer than any temperature
/// difference that changes a probability in the third decimal.
const GOLDEN_SECTION_STEPS: usize = 60;

/// The temperature in the applied range that best explains the right
/// answers, or `None` for no questions.
#[must_use]
pub fn fit_temperature(scored_questions: &[&ScoredQuestion]) -> Option<f64> {
    if scored_questions.is_empty() {
        return None;
    }
    let mut lower_log_temperature = MINIMUM_APPLIED_TEMPERATURE.ln();
    let mut upper_log_temperature = MAXIMUM_APPLIED_TEMPERATURE.ln();
    for _ in 0..GOLDEN_SECTION_STEPS {
        let bracket_width = upper_log_temperature - lower_log_temperature;
        let left_probe = upper_log_temperature - GOLDEN_RATIO_CONJUGATE * bracket_width;
        let right_probe = lower_log_temperature + GOLDEN_RATIO_CONJUGATE * bracket_width;
        let left_loss = mean_negative_log_likelihood(scored_questions, left_probe.exp());
        let right_loss = mean_negative_log_likelihood(scored_questions, right_probe.exp());
        if left_loss <= right_loss {
            upper_log_temperature = right_probe;
        } else {
            lower_log_temperature = left_probe;
        }
    }
    Some(((lower_log_temperature + upper_log_temperature) / 2.0).exp())
}

/// The mean negative log-likelihood of the right answers at one temperature.
#[must_use]
pub fn mean_negative_log_likelihood(scored_questions: &[&ScoredQuestion], temperature: f64) -> f64 {
    let mut negative_log_likelihood_total = 0.0;
    for scored_question in scored_questions {
        let log_probabilities = log_probabilities_at(&scored_question.option_logits, temperature);
        negative_log_likelihood_total -= log_probabilities[scored_question.correct_option_index];
    }
    negative_log_likelihood_total / scored_questions.len() as f64
}

#[cfg(test)]
#[path = "temperature_fit_tests.rs"]
mod tests;
