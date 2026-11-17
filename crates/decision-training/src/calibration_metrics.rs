//! How good a set of probabilities is against the right answers.
//!
//! Four numbers, on the questions a fit is judged on:
//!
//! * accuracy: how often the most probable option is the right one. A
//!   temperature never changes which option is most probable, so this must
//!   be identical before and after calibration; the plan checks it.
//! * negative log-likelihood: the average surprise at the right answer. It
//!   is what the temperature is fitted to lower, and what the judgement compares.
//! * Brier score: the squared distance between the probabilities and the
//!   right answer, a second view that punishes confident mistakes less.
//! * expected calibration error: the gap between stated confidence and
//!   observed accuracy. Its value depends on the binning, so the binning is
//!   fixed and stated here: 15 equal-width bins of the top probability,
//!   weighted by how many questions fall in each.

use astronomical_decision_contract::calibration::clamp_temperature;

use crate::scored_question::ScoredQuestion;
use crate::temperature_softmax::{first_largest_index, log_probabilities_at};

/// Equal-width confidence bins for the expected calibration error.
pub const CALIBRATION_ERROR_BIN_COUNT: usize = 15;

/// Which temperature each question is scored at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoringTemperature<'calibration> {
    /// What the model applied when it served the answer.
    Shipped,
    /// A kind calibration, applied as an engine applies it.
    Calibrated(&'calibration astronomical_decision_contract::calibration_file::KindCalibration),
}

impl ScoringTemperature<'_> {
    /// The temperature `scored_question` is scored at.
    #[must_use]
    pub fn temperature_for(&self, scored_question: &ScoredQuestion) -> f64 {
        match self {
            // As served: Kev, for one, applies its shipped value unclamped.
            Self::Shipped => scored_question.shipped_temperature,
            Self::Calibrated(kind_calibration) => clamp_temperature(
                kind_calibration.temperature_for_option_count(scored_question.option_count()),
            ),
        }
    }
}

/// The four quality numbers over a set of questions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CalibrationMetrics {
    pub question_count: usize,
    pub correct_count: usize,
    pub mean_negative_log_likelihood: f64,
    pub mean_brier_score: f64,
    pub expected_calibration_error: f64,
}

impl CalibrationMetrics {
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        if self.question_count == 0 {
            return 0.0;
        }
        self.correct_count as f64 / self.question_count as f64
    }
}

/// The metrics of `scored_questions` at `scoring_temperature`.
#[must_use]
pub fn calibration_metrics(
    scored_questions: &[&ScoredQuestion],
    scoring_temperature: ScoringTemperature<'_>,
) -> CalibrationMetrics {
    let mut metrics = CalibrationMetrics {
        question_count: scored_questions.len(),
        ..CalibrationMetrics::default()
    };
    if scored_questions.is_empty() {
        return metrics;
    }
    let mut confidence_bins = [ConfidenceBin::default(); CALIBRATION_ERROR_BIN_COUNT];
    let mut negative_log_likelihood_total = 0.0;
    let mut brier_score_total = 0.0;
    for scored_question in scored_questions {
        let temperature = scoring_temperature.temperature_for(scored_question);
        let log_probabilities = log_probabilities_at(&scored_question.option_logits, temperature);
        let correct_index = scored_question.correct_option_index;
        negative_log_likelihood_total -= log_probabilities[correct_index];
        brier_score_total += brier_score(&log_probabilities, correct_index);
        let chosen_index = first_largest_index(&log_probabilities);
        let is_correct = chosen_index == correct_index;
        metrics.correct_count += usize::from(is_correct);
        let top_probability = log_probabilities[chosen_index].exp();
        confidence_bins[confidence_bin_index(top_probability)].add(top_probability, is_correct);
    }
    let question_count = scored_questions.len() as f64;
    metrics.mean_negative_log_likelihood = negative_log_likelihood_total / question_count;
    metrics.mean_brier_score = brier_score_total / question_count;
    metrics.expected_calibration_error =
        expected_calibration_error(&confidence_bins, scored_questions.len());
    metrics
}

fn brier_score(log_probabilities: &[f64], correct_index: usize) -> f64 {
    let mut squared_distance = 0.0;
    for (option_index, log_probability) in log_probabilities.iter().enumerate() {
        let target = if option_index == correct_index {
            1.0
        } else {
            0.0
        };
        squared_distance += (log_probability.exp() - target).powi(2);
    }
    squared_distance
}

#[derive(Clone, Copy, Debug, Default)]
struct ConfidenceBin {
    question_count: usize,
    confidence_total: f64,
    correct_count: usize,
}

impl ConfidenceBin {
    fn add(&mut self, top_probability: f64, is_correct: bool) {
        self.question_count += 1;
        self.confidence_total += top_probability;
        self.correct_count += usize::from(is_correct);
    }
}

fn confidence_bin_index(top_probability: f64) -> usize {
    let bin_index = (top_probability * CALIBRATION_ERROR_BIN_COUNT as f64) as usize;
    bin_index.min(CALIBRATION_ERROR_BIN_COUNT - 1)
}

fn expected_calibration_error(confidence_bins: &[ConfidenceBin], question_count: usize) -> f64 {
    let mut weighted_gap_total = 0.0;
    for confidence_bin in confidence_bins {
        if confidence_bin.question_count == 0 {
            continue;
        }
        let bin_size = confidence_bin.question_count as f64;
        let mean_confidence = confidence_bin.confidence_total / bin_size;
        let bin_accuracy = confidence_bin.correct_count as f64 / bin_size;
        weighted_gap_total += bin_size * (mean_confidence - bin_accuracy).abs();
    }
    weighted_gap_total / question_count as f64
}

#[cfg(test)]
#[path = "calibration_metrics_tests.rs"]
mod tests;
