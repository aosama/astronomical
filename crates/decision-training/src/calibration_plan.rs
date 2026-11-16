//! The never-worse gate: which question kinds get a fitted temperature.
//!
//! For each kind, temperatures are fitted on the fitting part of the split
//! and then judged on the held-out part against what the model shipped with.
//! A kind is calibrated only when the fit beats the shipped temperatures on
//! questions it never saw; otherwise the kind is left out of the calibration
//! file and keeps the family's own temperatures exactly. Too few questions
//! on either side also leaves a kind alone, because a temperature fitted or
//! judged on a handful of questions mostly measures noise. Laya's own
//! trainer refuses to fit below 10 held-out items for the same reason.
//!
//! Two candidates are tried per kind: one temperature for the whole kind,
//! and that plus a temperature per option-count range that has enough
//! fitting questions of its own. The per-range candidate is kept only when
//! it beats the single temperature on held-out questions too, so extra
//! numbers have to earn their place.
//!
//! The gate also checks the one thing a temperature can never do: change
//! which option is most probable. If it ever does, the scoring code is wrong
//! and nothing is written.

use astronomical_decision_contract::DecisionQuestionKind;
use astronomical_decision_contract::calibration::{
    OPTION_COUNT_RANGE_NAMES, option_count_range_name,
};
use astronomical_decision_contract::calibration_file::{CalibrationOverride, KindCalibration};
use thiserror::Error;

use crate::calibration_metrics::{CalibrationMetrics, ScoringTemperature, calibration_metrics};
use crate::holdout_split::HoldoutSplit;
use crate::scored_question::ScoredQuestion;
use crate::temperature_fit::fit_temperature;
use crate::temperature_softmax::{first_largest_index, log_probabilities_at};

/// Fewest fitting questions a kind needs before a temperature is fitted.
pub const MINIMUM_FITTING_QUESTIONS_PER_KIND: usize = 20;
/// Fewest held-out questions a kind needs before a fit is judged.
pub const MINIMUM_HOLDOUT_QUESTIONS_PER_KIND: usize = 10;
/// Fewest fitting questions an option-count range needs for its own temperature.
pub const MINIMUM_FITTING_QUESTIONS_PER_RANGE: usize = 30;
/// How much lower, in nats per held-out question, the fitted loss must be
/// than the shipped loss before the fit is kept. A temperature that only
/// matches the shipped one to rounding error is not an improvement, and
/// writing it would make a "calibrated" model that is the same model.
pub const MINIMUM_HOLDOUT_LOSS_IMPROVEMENT: f64 = 1e-4;
/// Log-probabilities closer than this are a tie, which may break either way.
const TIE_LOG_PROBABILITY_GAP: f64 = 1e-12;

const ALL_QUESTION_KINDS: [DecisionQuestionKind; 3] = [
    DecisionQuestionKind::Choice,
    DecisionQuestionKind::Score,
    DecisionQuestionKind::Noul,
];

/// What the gate decided for every kind, and the file that records it.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationPlan {
    pub kind_verdicts: Vec<KindVerdict>,
    pub calibration_override: CalibrationOverride,
}

/// What the gate decided for one kind.
#[derive(Clone, Debug, PartialEq)]
pub struct KindVerdict {
    pub kind: DecisionQuestionKind,
    pub fitting_question_count: usize,
    pub holdout_question_count: usize,
    /// The shipped temperatures on held-out questions, when there were any.
    pub shipped_holdout_metrics: Option<CalibrationMetrics>,
    pub outcome: KindOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub enum KindOutcome {
    Calibrated {
        kind_calibration: KindCalibration,
        calibrated_holdout_metrics: CalibrationMetrics,
    },
    KeptShipped(KeptShippedReason),
}

/// Why a kind keeps the temperatures it shipped with.
#[derive(Clone, Debug, PartialEq)]
pub enum KeptShippedReason {
    NoQuestions,
    TooFewFittingQuestions {
        have: usize,
        need: usize,
    },
    TooFewHoldoutQuestions {
        have: usize,
        need: usize,
    },
    NoImprovement {
        best_holdout_metrics: CalibrationMetrics,
    },
}

/// The scoring code broke an invariant, so no calibration can be trusted.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum CalibrationPlanError {
    #[error(
        "a temperature changed which answer is most probable for question {question:?}, which a temperature cannot do; calibration stopped and nothing was written"
    )]
    TemperatureChangedTheAnswer { question: String },
}

/// Runs the gate over every kind.
pub fn plan_calibration(
    holdout_split: &HoldoutSplit,
) -> Result<CalibrationPlan, CalibrationPlanError> {
    let mut calibration_plan = CalibrationPlan {
        kind_verdicts: Vec::with_capacity(ALL_QUESTION_KINDS.len()),
        calibration_override: CalibrationOverride::default(),
    };
    for kind in ALL_QUESTION_KINDS {
        let kind_verdict = plan_kind(kind, holdout_split)?;
        if let KindOutcome::Calibrated {
            kind_calibration, ..
        } = &kind_verdict.outcome
        {
            calibration_plan
                .calibration_override
                .set_kind_calibration(kind, kind_calibration.clone());
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
    let shipped_holdout_metrics = if holdout_questions.is_empty() {
        None
    } else {
        Some(calibration_metrics(
            &holdout_questions,
            ScoringTemperature::Shipped,
        ))
    };
    let mut kind_verdict = KindVerdict {
        kind,
        fitting_question_count: fitting_questions.len(),
        holdout_question_count: holdout_questions.len(),
        shipped_holdout_metrics,
        outcome: KindOutcome::KeptShipped(KeptShippedReason::NoQuestions),
    };
    if let Some(too_few_reason) = too_few_questions_reason(&fitting_questions, &holdout_questions) {
        kind_verdict.outcome = KindOutcome::KeptShipped(too_few_reason);
        return Ok(kind_verdict);
    }
    let best_candidate = best_fitted_candidate(&fitting_questions, &holdout_questions);
    check_answers_unchanged(&holdout_questions, &best_candidate.kind_calibration)?;
    // Too-few-questions has already returned, so held-out metrics exist here.
    let shipped_loss = match shipped_holdout_metrics {
        Some(shipped_metrics) => shipped_metrics.mean_negative_log_likelihood,
        None => f64::INFINITY,
    };
    let calibrated_loss = best_candidate.holdout_metrics.mean_negative_log_likelihood;
    kind_verdict.outcome = if calibrated_loss < shipped_loss - MINIMUM_HOLDOUT_LOSS_IMPROVEMENT {
        KindOutcome::Calibrated {
            kind_calibration: best_candidate.kind_calibration,
            calibrated_holdout_metrics: best_candidate.holdout_metrics,
        }
    } else {
        KindOutcome::KeptShipped(KeptShippedReason::NoImprovement {
            best_holdout_metrics: best_candidate.holdout_metrics,
        })
    };
    Ok(kind_verdict)
}

fn too_few_questions_reason(
    fitting_questions: &[&ScoredQuestion],
    holdout_questions: &[&ScoredQuestion],
) -> Option<KeptShippedReason> {
    if fitting_questions.is_empty() && holdout_questions.is_empty() {
        return Some(KeptShippedReason::NoQuestions);
    }
    if fitting_questions.len() < MINIMUM_FITTING_QUESTIONS_PER_KIND {
        return Some(KeptShippedReason::TooFewFittingQuestions {
            have: fitting_questions.len(),
            need: MINIMUM_FITTING_QUESTIONS_PER_KIND,
        });
    }
    if holdout_questions.len() < MINIMUM_HOLDOUT_QUESTIONS_PER_KIND {
        return Some(KeptShippedReason::TooFewHoldoutQuestions {
            have: holdout_questions.len(),
            need: MINIMUM_HOLDOUT_QUESTIONS_PER_KIND,
        });
    }
    None
}

struct FittedCandidate {
    kind_calibration: KindCalibration,
    holdout_metrics: CalibrationMetrics,
}

/// The single-temperature candidate, or the per-range one when it does
/// strictly better on held-out questions.
fn best_fitted_candidate(
    fitting_questions: &[&ScoredQuestion],
    holdout_questions: &[&ScoredQuestion],
) -> FittedCandidate {
    let kind_temperature = fit_temperature(fitting_questions).unwrap_or(1.0);
    let single_temperature = KindCalibration {
        temperature: kind_temperature,
        range_temperatures: Vec::new(),
        fitted_question_count: fitting_questions.len(),
    };
    let single_candidate = judged_candidate(single_temperature, holdout_questions);
    let range_temperatures = fitted_range_temperatures(fitting_questions);
    if range_temperatures.is_empty() {
        return single_candidate;
    }
    let per_range = KindCalibration {
        range_temperatures,
        ..single_candidate.kind_calibration.clone()
    };
    let per_range_candidate = judged_candidate(per_range, holdout_questions);
    if per_range_candidate
        .holdout_metrics
        .mean_negative_log_likelihood
        < single_candidate
            .holdout_metrics
            .mean_negative_log_likelihood
    {
        return per_range_candidate;
    }
    single_candidate
}

fn judged_candidate(
    kind_calibration: KindCalibration,
    holdout_questions: &[&ScoredQuestion],
) -> FittedCandidate {
    let holdout_metrics = calibration_metrics(
        holdout_questions,
        ScoringTemperature::Calibrated(&kind_calibration),
    );
    FittedCandidate {
        kind_calibration,
        holdout_metrics,
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
        if range_questions.len() < MINIMUM_FITTING_QUESTIONS_PER_RANGE {
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

/// Refuses a calibration under which any held-out question's most probable
/// option differs from the one it was served, ties aside.
fn check_answers_unchanged(
    holdout_questions: &[&ScoredQuestion],
    kind_calibration: &KindCalibration,
) -> Result<(), CalibrationPlanError> {
    let calibrated = ScoringTemperature::Calibrated(kind_calibration);
    for scored_question in holdout_questions {
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
