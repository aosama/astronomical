//! Fitting temperatures per question kind, and judging each fit.
//!
//! For each kind, temperatures are fitted on the fitting part of the split
//! and judged on the held-out part against what the model shipped with. Any
//! number of questions is fitted: every kind the examples ask is calibrated,
//! and the judgement says whether it helped. A kind with no held-out
//! questions (or none to fit on) is fitted on all of its questions and
//! judged on them too, which the plan records as not judged. Nothing here
//! refuses; the report and the folder's training record carry the findings
//! and the person decides.
//!
//! Two candidates are tried per kind: one temperature for the whole kind,
//! and that plus a temperature per option-count range. A range gets its own
//! temperature only with [`RANGE_TEMPERATURE_QUESTION_COUNT`] fitting
//! questions; that decides how many numbers are fitted, never whether the
//! kind is calibrated. The per-range candidate is kept only when it beats
//! the single temperature on the judged questions, so extra numbers have to
//! earn their place.
//!
//! One check stays a hard stop, because no person's choice is involved: a
//! temperature can never change which option is most probable. If one ever
//! does, the scoring code is wrong and nothing is written.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration::{
    OPTION_COUNT_RANGE_NAMES, option_count_range_name,
};
use astronomical_decision_contract::calibration_file::{CalibrationOverride, KindCalibration};
use astronomical_decision_contract::training_record::RecordedVerdict;
use thiserror::Error;

use crate::calibration_metrics::{CalibrationMetrics, ScoringTemperature, calibration_metrics};
use crate::holdout_split::HoldoutSplit;
use crate::judged_on::JudgedOn;
use crate::scored_question::ScoredQuestion;
use crate::temperature_fit::fit_temperature;
use crate::temperature_softmax::{first_largest_index, log_probabilities_at};

/// Fitting questions an option-count range needs for a temperature of its
/// own; with fewer, the range uses its kind's single temperature. A
/// temperature fitted on a few questions mostly measures noise.
pub const RANGE_TEMPERATURE_QUESTION_COUNT: usize = 30;
/// How much lower, in nats per judged question, the fitted loss must be than
/// the shipped loss to count as an improvement. A temperature that only
/// matches the shipped one to rounding error is not one.
pub const MINIMUM_LOSS_IMPROVEMENT: f64 = 1e-4;
/// Log-probabilities closer than this are a tie, which may break either way.
const TIE_LOG_PROBABILITY_GAP: f64 = 1e-12;

const ALL_QUESTION_KINDS: [DecisionQuestionKind; 3] = [
    DecisionQuestionKind::Choice,
    DecisionQuestionKind::Score,
    DecisionQuestionKind::Noul,
];

/// Every kind's fit and judgement, and the file that records the fits.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationPlan {
    pub kind_verdicts: Vec<KindVerdict>,
    pub calibration_override: CalibrationOverride,
}

impl CalibrationPlan {
    /// What the folder's training record says: not judged when any kind
    /// could only be compared on the questions it was fitted to, judged
    /// worse when any judged kind did not improve, otherwise judged better.
    #[must_use]
    pub fn recorded_verdict(&self) -> RecordedVerdict {
        let mut any_unjudged = false;
        let mut any_not_improved = false;
        for kind_verdict in &self.kind_verdicts {
            if let KindOutcome::Calibrated(kind_fit) = &kind_verdict.outcome {
                any_unjudged |= kind_fit.judged_on == JudgedOn::LearnedFromExamples;
                any_not_improved |= !kind_fit.improved();
            }
        }
        if any_unjudged {
            RecordedVerdict::NotJudged
        } else if any_not_improved {
            RecordedVerdict::JudgedWorse
        } else {
            RecordedVerdict::JudgedBetter
        }
    }
}

/// One kind's fit and what judging it showed.
#[derive(Clone, Debug, PartialEq)]
pub struct KindVerdict {
    pub kind: DecisionQuestionKind,
    pub fitting_question_count: usize,
    pub holdout_question_count: usize,
    pub outcome: KindOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KindOutcome {
    Calibrated(KindFit),
    /// The examples ask no question of this kind.
    NoQuestions,
}

/// A fitted kind, compared with the shipped temperatures.
#[derive(Clone, Debug, PartialEq)]
pub struct KindFit {
    pub kind_calibration: KindCalibration,
    pub judged_on: JudgedOn,
    /// The shipped temperatures on the judged questions.
    pub shipped_metrics: CalibrationMetrics,
    /// The fitted temperatures on the same questions.
    pub calibrated_metrics: CalibrationMetrics,
}

impl KindFit {
    /// Whether the fit was less surprised by the right answers than the
    /// shipped temperatures, beyond rounding.
    #[must_use]
    pub fn improved(&self) -> bool {
        self.calibrated_metrics.mean_negative_log_likelihood
            < self.shipped_metrics.mean_negative_log_likelihood - MINIMUM_LOSS_IMPROVEMENT
    }
}

/// The scoring code broke an invariant, so no calibration can be trusted.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CalibrationPlanError {
    #[error(
        "a temperature changed which answer is most probable for question {question:?}, which a temperature cannot do; calibration stopped and nothing was written"
    )]
    TemperatureChangedTheAnswer { question: String },
}

/// Fits and judges every kind the examples ask.
pub fn plan_calibration(
    holdout_split: &HoldoutSplit,
) -> Result<CalibrationPlan, CalibrationPlanError> {
    let mut calibration_plan = CalibrationPlan {
        kind_verdicts: Vec::with_capacity(ALL_QUESTION_KINDS.len()),
        calibration_override: CalibrationOverride::default(),
    };
    for kind in ALL_QUESTION_KINDS {
        let kind_verdict = plan_kind(kind, holdout_split)?;
        if let KindOutcome::Calibrated(kind_fit) = &kind_verdict.outcome {
            calibration_plan
                .calibration_override
                .set_kind_calibration(kind, kind_fit.kind_calibration.clone());
        }
        calibration_plan.kind_verdicts.push(kind_verdict);
    }
    Ok(calibration_plan)
}

fn plan_kind(
    kind: DecisionQuestionKind,
    holdout_split: &HoldoutSplit,
) -> Result<KindVerdict, CalibrationPlanError> {
    let fitting_questions = questions_of_kind(&holdout_split.fitting_questions, kind);
    let holdout_questions = questions_of_kind(&holdout_split.holdout_questions, kind);
    let mut kind_verdict = KindVerdict {
        kind,
        fitting_question_count: fitting_questions.len(),
        holdout_question_count: holdout_questions.len(),
        outcome: KindOutcome::NoQuestions,
    };
    if fitting_questions.is_empty() && holdout_questions.is_empty() {
        return Ok(kind_verdict);
    }
    // Without both sides there is nothing to hold back: fit on every
    // question of the kind and compare on the same ones.
    let (fitted, judged, judged_on) =
        if fitting_questions.is_empty() || holdout_questions.is_empty() {
            let mut every_question = fitting_questions;
            every_question.extend(holdout_questions);
            (
                every_question.clone(),
                every_question,
                JudgedOn::LearnedFromExamples,
            )
        } else {
            (
                fitting_questions,
                holdout_questions,
                JudgedOn::HeldBackQuestions,
            )
        };
    let best_candidate = best_fitted_candidate(&fitted, &judged);
    check_answers_unchanged(&judged, &best_candidate.kind_calibration)?;
    kind_verdict.outcome = KindOutcome::Calibrated(KindFit {
        kind_calibration: best_candidate.kind_calibration,
        judged_on,
        shipped_metrics: calibration_metrics(&judged, ScoringTemperature::Shipped),
        calibrated_metrics: best_candidate.judged_metrics,
    });
    Ok(kind_verdict)
}

struct FittedCandidate {
    kind_calibration: KindCalibration,
    judged_metrics: CalibrationMetrics,
}

/// The single-temperature candidate, or the per-range one when it does
/// strictly better on the judged questions.
fn best_fitted_candidate(
    fitting_questions: &[&ScoredQuestion],
    judged_questions: &[&ScoredQuestion],
) -> FittedCandidate {
    let kind_temperature = fit_temperature(fitting_questions).unwrap_or(1.0);
    let single_temperature = KindCalibration {
        temperature: kind_temperature,
        range_temperatures: Vec::new(),
        fitted_question_count: fitting_questions.len(),
    };
    let single_candidate = judged_candidate(single_temperature, judged_questions);
    let range_temperatures = fitted_range_temperatures(fitting_questions);
    if range_temperatures.is_empty() {
        return single_candidate;
    }
    let per_range = KindCalibration {
        range_temperatures,
        ..single_candidate.kind_calibration.clone()
    };
    let per_range_candidate = judged_candidate(per_range, judged_questions);
    if per_range_candidate
        .judged_metrics
        .mean_negative_log_likelihood
        < single_candidate.judged_metrics.mean_negative_log_likelihood
    {
        return per_range_candidate;
    }
    single_candidate
}

fn judged_candidate(
    kind_calibration: KindCalibration,
    judged_questions: &[&ScoredQuestion],
) -> FittedCandidate {
    let judged_metrics = calibration_metrics(
        judged_questions,
        ScoringTemperature::Calibrated(&kind_calibration),
    );
    FittedCandidate {
        kind_calibration,
        judged_metrics,
    }
}

fn fitted_range_temperatures(fitting_questions: &[&ScoredQuestion]) -> Vec<(String, f64)> {
    let mut range_temperatures = Vec::new();
    for range_name in OPTION_COUNT_RANGE_NAMES {
        let mut range_questions = Vec::new();
        for scored_question in fitting_questions {
            if option_count_range_name(scored_question.option_count()) == range_name {
                range_questions.push(*scored_question);
            }
        }
        if range_questions.len() < RANGE_TEMPERATURE_QUESTION_COUNT {
            continue;
        }
        if let Some(range_temperature) = fit_temperature(&range_questions) {
            range_temperatures.push((range_name.to_owned(), range_temperature));
        }
    }
    range_temperatures
}

fn questions_of_kind(
    scored_questions: &[ScoredQuestion],
    kind: DecisionQuestionKind,
) -> Vec<&ScoredQuestion> {
    let mut kind_questions = Vec::new();
    for scored_question in scored_questions {
        if scored_question.kind == kind {
            kind_questions.push(scored_question);
        }
    }
    kind_questions
}

/// Stops a calibration under which any judged question's most probable
/// option differs from the one it was served, ties aside.
fn check_answers_unchanged(
    judged_questions: &[&ScoredQuestion],
    kind_calibration: &KindCalibration,
) -> Result<(), CalibrationPlanError> {
    let calibrated = ScoringTemperature::Calibrated(kind_calibration);
    for scored_question in judged_questions {
        let shipped_log_probabilities = log_probabilities_at(
            &scored_question.option_logits,
            scored_question.shipped_temperature,
        );
        let calibrated_log_probabilities = log_probabilities_at(
            &scored_question.option_logits,
            calibrated.temperature_for(scored_question),
        );
        let shipped_choice = first_largest_index(&shipped_log_probabilities);
        let calibrated_choice = first_largest_index(&calibrated_log_probabilities);
        let is_tie = (calibrated_log_probabilities[shipped_choice]
            - calibrated_log_probabilities[calibrated_choice])
            .abs()
            <= TIE_LOG_PROBABILITY_GAP;
        if shipped_choice != calibrated_choice && !is_tie {
            return Err(CalibrationPlanError::TemperatureChangedTheAnswer {
                question: scored_question.question_name.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "calibration_plan_tests.rs"]
mod tests;
