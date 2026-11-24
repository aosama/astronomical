//! Judging a trained head against the shipped model.
//!
//! The trained model is the trained head with its own temperatures, fitted
//! and chosen on the choosing questions alone. It is compared on the judging
//! questions, which nothing before this point has read, with the shipped
//! head given the same treatment: its own temperatures fitted the same way
//! on the same choosing questions. Comparing with the shipped model as
//! served would let training take the credit for what calibration alone
//! does, and a head that training made worse could look better behind a
//! temperature that made up for it.
//!
//! The judgement never decides for the person: the folder is written
//! whatever it finds, and the findings go into the report and the folder's
//! training record. It is judged better only when none of these concerns
//! appear:
//!
//! - it is right less often than the shipped head
//! - it is not less surprised by the right answers overall than the
//!   calibrated shipped head
//! - a question kind got more surprised
//! - it changed the shipped answer on more than 5% of the built-in check
//!   set, the questions nobody labelled
//! - it collapsed into giving one answer to every question of a kind whose
//!   labels vary (the failure Laya's training notes warn small data produces)
//!
//! With too few different texts to hold any back, the comparison runs on the
//! training examples themselves; the verdict is then "not judged", since
//! doing well on what it learned from says nothing about new questions.

use decision_contract::DecisionQuestionKind;
use decision_contract::calibration_file::CalibrationOverride;
use decision_contract::training_record::RecordedVerdict;

use crate::calibration_metrics::{CalibrationMetrics, ScoringTemperature, calibration_metrics};
use crate::calibration_plan::{
    CalibrationPlan, CalibrationPlanError, KindOutcome, plan_calibration,
};
use crate::holdout_split::HoldoutSplit;
pub use crate::judged_on::JudgedOn;
use crate::scored_question::ScoredQuestion;
use crate::temperature_softmax::first_largest_index;

/// The share of check-set answers below which a changed check set is a concern.
pub const MINIMUM_CHECK_AGREEMENT: f64 = 0.95;
/// One answer this common among a kind's judging questions is a collapse,
/// unless the labels themselves are that uniform.
const COLLAPSE_SHARE: f64 = 0.95;
/// Lower surprise by less than this is rounding, not improvement.
const MINIMUM_SURPRISE_IMPROVEMENT: f64 = 1e-4;

const ALL_QUESTION_KINDS: [DecisionQuestionKind; 3] = [
    DecisionQuestionKind::Choice,
    DecisionQuestionKind::Score,
    DecisionQuestionKind::Noul,
];

/// Shipped and trained, side by side, for one kind's judging questions.
#[derive(Clone, Debug, PartialEq)]
pub struct KindComparison {
    pub kind: DecisionQuestionKind,
    pub shipped: CalibrationMetrics,
    pub trained: CalibrationMetrics,
}

/// How many check-set answers stayed as shipped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckAgreement {
    pub agreeing_count: usize,
    pub question_count: usize,
}

/// What the comparison found that a person should weigh before using the
/// trained model.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingConcern {
    LessAccurate {
        shipped: f64,
        trained: f64,
    },
    NotLessSurprised {
        shipped: f64,
        trained: f64,
    },
    KindMoreSurprised {
        kind: DecisionQuestionKind,
        /// How many judged questions of the kind this rests on.
        question_count: usize,
        shipped: f64,
        trained: f64,
    },
    CheckAnswersChanged(CheckAgreement),
    Collapsed {
        kind: DecisionQuestionKind,
    },
}

/// The findings about one trained head.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingVerdict {
    /// The shipped model exactly as it serves today.
    pub served_overall: CalibrationMetrics,
    /// The shipped head with its own fitted temperatures: what calibration
    /// alone would give, and what training must beat.
    pub shipped_overall: CalibrationMetrics,
    pub trained_overall: CalibrationMetrics,
    pub kind_comparisons: Vec<KindComparison>,
    pub check_agreement: CheckAgreement,
    /// The trained model's temperatures, written beside its head.
    pub calibration_plan: CalibrationPlan,
    pub judged_on: JudgedOn,
    pub concerns: Vec<TrainingConcern>,
}

impl TrainingVerdict {
    /// What the folder's training record says about this head.
    #[must_use]
    pub fn recorded_verdict(&self) -> RecordedVerdict {
        // Every judged question can be left out because its options read
        // alike; a comparison over none says nothing either way.
        if self.shipped_overall.question_count == 0 {
            return RecordedVerdict::NotJudged;
        }
        match (self.judged_on, self.concerns.is_empty()) {
            (JudgedOn::LearnedFromExamples, _) => RecordedVerdict::NotJudged,
            (JudgedOn::HeldBackQuestions, true) => RecordedVerdict::JudgedBetter,
            (JudgedOn::HeldBackQuestions, false) => RecordedVerdict::JudgedNotBetter,
        }
    }

    #[must_use]
    pub const fn calibration_override(&self) -> &CalibrationOverride {
        &self.calibration_plan.calibration_override
    }
}

/// One labelled question as the shipped model served it, and the trained
/// head's scores for it in the same option order.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredPair {
    pub served: ScoredQuestion,
    pub trained_logits: Vec<f32>,
}

/// One check-set question: the probabilities the shipped model served and
/// the trained head's scores.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckPair {
    pub served_probabilities: Vec<f64>,
    pub trained_logits: Vec<f32>,
}

/// Everything the judgement reads about one training.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingScores {
    pub choosing: Vec<ScoredPair>,
    pub judging: Vec<ScoredPair>,
    pub check: Vec<CheckPair>,
    /// What `judging` holds: held-back questions, or, when nothing was held
    /// back, the training examples (and `choosing` holds them too).
    pub judged_on: JudgedOn,
}

/// Fits the trained model's temperatures and compares it with the shipped model.
pub fn judge_training(
    training_scores: &TrainingScores,
) -> Result<TrainingVerdict, CalibrationPlanError> {
    let calibration_plan = choosing_calibration(trained_questions(&training_scores.choosing))?;
    let shipped_plan = choosing_calibration(served_questions(&training_scores.choosing))?;
    let served_judging = served_questions(&training_scores.judging);
    let shipped_judging = as_served_after_calibration(served_judging.clone(), &shipped_plan);
    let trained_judging = as_served_after_calibration(
        trained_questions(&training_scores.judging),
        &calibration_plan,
    );
    let mut verdict = TrainingVerdict {
        served_overall: metrics_of(&served_judging, None),
        shipped_overall: metrics_of(&shipped_judging, None),
        trained_overall: metrics_of(&trained_judging, None),
        kind_comparisons: kind_comparisons(&shipped_judging, &trained_judging),
        check_agreement: check_agreement(&training_scores.check),
        calibration_plan,
        judged_on: training_scores.judged_on,
        concerns: Vec::new(),
    };
    verdict.concerns = concerns(&verdict, &trained_judging);
    Ok(verdict)
}

/// Temperatures fitted and chosen on the choosing questions alone, so the
/// judging questions stay unread until the comparison.
fn choosing_calibration(
    choosing_questions: Vec<ScoredQuestion>,
) -> Result<CalibrationPlan, CalibrationPlanError> {
    plan_calibration(&HoldoutSplit {
        fitting_questions: choosing_questions.clone(),
        holdout_questions: choosing_questions,
    })
}

/// The questions with the trained head's scores in place of the served ones.
fn trained_questions(scored_pairs: &[ScoredPair]) -> Vec<ScoredQuestion> {
    let mut trained = Vec::with_capacity(scored_pairs.len());
    for scored_pair in scored_pairs {
        let mut scored_question = scored_pair.served.clone();
        scored_question
            .option_logits
            .clone_from(&scored_pair.trained_logits);
        trained.push(scored_question);
    }
    trained
}

fn served_questions(scored_pairs: &[ScoredPair]) -> Vec<ScoredQuestion> {
    let mut served = Vec::with_capacity(scored_pairs.len());
    for scored_pair in scored_pairs {
        served.push(scored_pair.served.clone());
    }
    served
}

/// Bakes each question's temperature, as the trained folder will serve it,
/// into the question, so one scoring pass reads every kind correctly.
fn as_served_after_calibration(
    mut scored_questions: Vec<ScoredQuestion>,
    calibration_plan: &CalibrationPlan,
) -> Vec<ScoredQuestion> {
    for scored_question in &mut scored_questions {
        for kind_verdict in &calibration_plan.kind_verdicts {
            if let KindOutcome::Calibrated(kind_fit) = &kind_verdict.outcome
                && kind_verdict.kind == scored_question.kind
            {
                scored_question.shipped_temperature =
                    ScoringTemperature::Calibrated(&kind_fit.kind_calibration)
                        .temperature_for(scored_question);
            }
        }
    }
    scored_questions
}

fn metrics_of(
    scored_questions: &[ScoredQuestion],
    kind: Option<DecisionQuestionKind>,
) -> CalibrationMetrics {
    let mut selected = Vec::with_capacity(scored_questions.len());
    for scored_question in scored_questions {
        if kind.is_none() || kind == Some(scored_question.kind) {
            selected.push(scored_question);
        }
    }
    calibration_metrics(&selected, ScoringTemperature::Shipped)
}

fn kind_comparisons(shipped: &[ScoredQuestion], trained: &[ScoredQuestion]) -> Vec<KindComparison> {
    let mut comparisons = Vec::new();
    for kind in ALL_QUESTION_KINDS {
        let shipped_metrics = metrics_of(shipped, Some(kind));
        if shipped_metrics.question_count > 0 {
            comparisons.push(KindComparison {
                kind,
                shipped: shipped_metrics,
                trained: metrics_of(trained, Some(kind)),
            });
        }
    }
    comparisons
}

fn check_agreement(check_pairs: &[CheckPair]) -> CheckAgreement {
    let mut agreeing_count = 0;
    for check_pair in check_pairs {
        let shipped_answer = first_largest_index(&check_pair.served_probabilities);
        agreeing_count +=
            usize::from(first_largest_logit_index(&check_pair.trained_logits) == shipped_answer);
    }
    CheckAgreement {
        agreeing_count,
        question_count: check_pairs.len(),
    }
}

/// The first option with the largest logit: the answer a model gives.
fn first_largest_logit_index(option_logits: &[f32]) -> usize {
    let mut largest_index = 0;
    for (option_index, option_logit) in option_logits.iter().enumerate() {
        if *option_logit > option_logits[largest_index] {
            largest_index = option_index;
        }
    }
    largest_index
}

fn concerns(verdict: &TrainingVerdict, trained_judging: &[ScoredQuestion]) -> Vec<TrainingConcern> {
    let mut concerns = Vec::new();
    let (shipped, trained) = (&verdict.shipped_overall, &verdict.trained_overall);
    if shipped.question_count > 0 {
        if trained.correct_count < shipped.correct_count {
            concerns.push(TrainingConcern::LessAccurate {
                shipped: shipped.accuracy(),
                trained: trained.accuracy(),
            });
        }
        if trained.mean_negative_log_likelihood
            > shipped.mean_negative_log_likelihood - MINIMUM_SURPRISE_IMPROVEMENT
        {
            concerns.push(TrainingConcern::NotLessSurprised {
                shipped: shipped.mean_negative_log_likelihood,
                trained: trained.mean_negative_log_likelihood,
            });
        }
    }
    // With a single kind its comparison repeats the overall one, so only a
    // mix of kinds is compared kind by kind.
    let compares_kinds = verdict.kind_comparisons.len() > 1;
    for comparison in &verdict.kind_comparisons {
        if compares_kinds
            && comparison.trained.mean_negative_log_likelihood
                > comparison.shipped.mean_negative_log_likelihood
        {
            concerns.push(TrainingConcern::KindMoreSurprised {
                kind: comparison.kind,
                question_count: comparison.shipped.question_count,
                shipped: comparison.shipped.mean_negative_log_likelihood,
                trained: comparison.trained.mean_negative_log_likelihood,
            });
        }
        if has_collapsed(trained_judging, comparison.kind) {
            concerns.push(TrainingConcern::Collapsed {
                kind: comparison.kind,
            });
        }
    }
    let agreement = verdict.check_agreement;
    if agreement.question_count > 0
        && (agreement.agreeing_count as f64)
            < MINIMUM_CHECK_AGREEMENT * agreement.question_count as f64
    {
        concerns.push(TrainingConcern::CheckAnswersChanged(agreement));
    }
    concerns
}

/// Whether one answer position covers nearly every judging question of
/// `kind` while the labels are spread out.
fn has_collapsed(trained_judging: &[ScoredQuestion], kind: DecisionQuestionKind) -> bool {
    let mut picked_counts = [0_usize; 64];
    let mut label_counts = [0_usize; 64];
    let mut question_count = 0;
    for scored_question in trained_judging {
        if scored_question.kind != kind {
            continue;
        }
        question_count += 1;
        picked_counts[first_largest_logit_index(&scored_question.option_logits).min(63)] += 1;
        label_counts[scored_question.correct_option_index.min(63)] += 1;
    }
    let most_picked = picked_counts.iter().max().copied().unwrap_or(0) as f64;
    let most_labelled = label_counts.iter().max().copied().unwrap_or(0) as f64;
    let threshold = COLLAPSE_SHARE * question_count as f64;
    question_count > 0 && most_picked >= threshold && most_labelled < threshold
}

#[cfg(test)]
#[path = "training_verdict_tests.rs"]
mod tests;
