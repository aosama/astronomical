//! Proving the written folder serves what was trained.
//!
//! The trained weights pass through a file, a reload and the engine's own
//! substitution before anyone uses them; Laya's own trainer once saved in a
//! lower precision than it trained in, and a test of the in-memory model
//! would never have noticed. So after writing, the judging examples are
//! asked again, of the new model through the daemon, and every answer must
//! match the scores the worker's trained head gave it: the same raw scores
//! within rounding, and the same temperature.

use std::collections::HashSet;

use astronomical_decision_contract::QuestionOutcome;
use astronomical_decision_training::calibration_metrics::ScoringTemperature;
use astronomical_decision_training::calibration_plan::{CalibrationPlan, KindOutcome};
use astronomical_decision_training::holdout_split::state_group_key;
use astronomical_decision_training::labelled_example::LabelledExample;
use astronomical_decision_training::scored_question::ScoredQuestion;
use astronomical_decision_training::training_verdict::ScoredPair;

use super::calibrate::example_requests;
use super::decision_tool::{DecisionToolDependencies, DecisionToolError};
use super::example_scoring::{ScoringPass, answer_examples};
use crate::DaemonProbe;

/// Temperatures that differ by more than this are different temperatures.
const TEMPERATURE_TOLERANCE: f64 = 1e-6;

/// A gap this small between the written model and the trained one is float
/// rounding: both run in 32-bit on the GPU, where MLX multiplies matrices
/// in TF32 and adds in a different order for a different batch shape.
/// Anything larger is a different computation.
const LOGIT_ABSOLUTE_TOLERANCE: f32 = 2e-3;
const LOGIT_RELATIVE_TOLERANCE: f32 = 1e-3;

/// What the written model must answer: each judging question's trained
/// scores, and the temperatures the gate fitted.
pub(super) struct TrainedExpectation<'expectation> {
    pub(super) calibration_plan: &'expectation CalibrationPlan,
    pub(super) judging_pairs: &'expectation [ScoredPair],
}

/// Asks the written model the judging examples and returns the largest
/// score gap, or why its answers differ from the trained model's.
pub(super) async fn check_saved_model(
    new_model_id: &str,
    labelled_examples: &[LabelledExample],
    trained_expectation: &TrainedExpectation<'_>,
    daemon_probe: &DaemonProbe,
    dependencies: &mut DecisionToolDependencies<'_>,
) -> Result<f32, DecisionToolError> {
    let mut judged_state_keys = HashSet::new();
    for judging_pair in trained_expectation.judging_pairs {
        judged_state_keys.insert(judging_pair.served.state_group_key);
    }
    let mut judged_examples = Vec::new();
    for labelled_example in labelled_examples {
        if judged_state_keys.contains(&state_group_key(&labelled_example.state)) {
            judged_examples.push(labelled_example.clone());
        }
    }
    let answered_examples = answer_examples(
        &example_requests(&judged_examples),
        &ScoringPass {
            decision_model_id: new_model_id,
            stage_name: "Checking the written model",
        },
        daemon_probe,
        dependencies,
    )
    .await?;
    let mut largest_gap = 0.0_f32;
    for (labelled_example, question_outcomes) in judged_examples.iter().zip(&answered_examples) {
        let state_key = state_group_key(&labelled_example.state);
        for question_outcome in question_outcomes {
            let gap = checked_answer(trained_expectation, state_key, question_outcome)?;
            largest_gap = largest_gap.max(gap);
        }
    }
    Ok(largest_gap)
}

/// One served answer against the trained model's; questions training left
/// out (their options read alike) have nothing to compare with.
fn checked_answer(
    trained_expectation: &TrainedExpectation<'_>,
    state_key: u64,
    question_outcome: &QuestionOutcome,
) -> Result<f32, DecisionToolError> {
    let Some(judging_pair) = judging_pair_for(
        trained_expectation.judging_pairs,
        state_key,
        &question_outcome.question_name,
    ) else {
        return Ok(0.0);
    };
    let logit_gap = largest_logit_gap(
        &judging_pair.trained_logits,
        &question_outcome.option_logits,
    );
    let expected_temperature =
        temperature_after_training(&judging_pair.served, trained_expectation.calibration_plan);
    let temperature_matches = (question_outcome.applied_temperature - expected_temperature).abs()
        <= TEMPERATURE_TOLERANCE;
    if !logit_gap.within_tolerance || !temperature_matches {
        return Err(DecisionToolError::SavedModelDisagrees {
            question_name: question_outcome.question_name.clone(),
            largest_gap: logit_gap.gap,
        });
    }
    Ok(logit_gap.gap)
}

fn judging_pair_for<'pairs>(
    judging_pairs: &'pairs [ScoredPair],
    state_key: u64,
    question_name: &str,
) -> Option<&'pairs ScoredPair> {
    for judging_pair in judging_pairs {
        let served = &judging_pair.served;
        if served.state_group_key == state_key && served.question_name == question_name {
            return Some(judging_pair);
        }
    }
    None
}

/// How far two sets of logits are apart, and whether that is only rounding.
pub(super) struct LogitGap {
    pub(super) gap: f32,
    pub(super) within_tolerance: bool,
}

/// The largest difference between matching logits; a length mismatch or a
/// non-finite logit is never within tolerance.
pub(super) fn largest_logit_gap(expected_logits: &[f32], served_logits: &[f32]) -> LogitGap {
    if expected_logits.len() != served_logits.len() {
        return LogitGap {
            gap: f32::INFINITY,
            within_tolerance: false,
        };
    }
    let mut logit_gap = LogitGap {
        gap: 0.0,
        within_tolerance: true,
    };
    for (expected_logit, served_logit) in expected_logits.iter().zip(served_logits) {
        let difference = (expected_logit - served_logit).abs();
        let allowed = LOGIT_ABSOLUTE_TOLERANCE + LOGIT_RELATIVE_TOLERANCE * served_logit.abs();
        let is_close = difference <= allowed;
        if !is_close {
            logit_gap.within_tolerance = false;
        }
        logit_gap.gap = logit_gap.gap.max(difference);
        if !difference.is_finite() {
            logit_gap.gap = f32::INFINITY;
        }
    }
    logit_gap
}

/// The temperature the trained folder serves a question with: its kind's
/// new calibration, or, for a kind training did not recalibrate, the
/// temperature it was served with before, which the folder keeps.
pub(super) fn temperature_after_training(
    scored_question: &ScoredQuestion,
    calibration_plan: &CalibrationPlan,
) -> f64 {
    for kind_verdict in &calibration_plan.kind_verdicts {
        if kind_verdict.kind != scored_question.kind {
            continue;
        }
        if let KindOutcome::Calibrated {
            kind_calibration, ..
        } = &kind_verdict.outcome
        {
            return ScoringTemperature::Calibrated(kind_calibration)
                .temperature_for(scored_question);
        }
    }
    scored_question.shipped_temperature
}

#[cfg(test)]
#[path = "saved_model_check_tests.rs"]
mod tests;
