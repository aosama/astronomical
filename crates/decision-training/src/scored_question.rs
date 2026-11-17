//! A labelled question together with what the model scored for it.
//!
//! Turning a model outcome into a scored question is where a mixup between
//! the label's option order and the model's option order would slip in
//! unnoticed (Clef, for one, scores `choice` options in alphabetical order
//! internally and maps them back). Every outcome is therefore checked before
//! it is used: it must answer the question it was asked, carry one finite
//! logit per option, and its served probabilities must be exactly what
//! `softmax(logits / applied temperature)` gives in canonical order. If they
//! are not, the logits are in a different order from the answer and any
//! temperature fitted on them would be fitted to the wrong options.

use astronomical_decision_contract::question::{NOUL_FALSE_INDEX, NOUL_TRUE_INDEX};
use astronomical_decision_contract::{DecisionAnswer, DecisionQuestionKind, QuestionOutcome};
use thiserror::Error;

use crate::labelled_example::LabelledQuestion;
use crate::temperature_softmax::probabilities_at;

/// How far a served probability may sit from the recomputed one. Engines
/// compute in `f64` from the same `f32` logits, so honest differences are
/// around 1e-7; an order mixup moves probabilities by far more than this.
const SERVED_PROBABILITY_TOLERANCE: f64 = 1e-4;

/// One labelled question and the model's raw scores for it.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredQuestion {
    pub question_name: String,
    pub kind: DecisionQuestionKind,
    /// Uncalibrated logits, in canonical option order.
    pub option_logits: Vec<f32>,
    /// The temperature the model applied when it served this answer.
    pub shipped_temperature: f64,
    pub correct_option_index: usize,
    /// Questions about the same state share this key, so they always land
    /// on the same side of the held-out split.
    pub state_group_key: u64,
}

impl ScoredQuestion {
    #[must_use]
    pub fn option_count(&self) -> usize {
        self.option_logits.len()
    }
}

/// Why a model outcome cannot be trusted for calibration.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScoredQuestionError {
    #[error("the model answered {answered:?} when {asked:?} was asked")]
    AnsweredAnotherQuestion { asked: String, answered: String },
    #[error("question {question:?} has {expected} options but the model returned {actual} scores")]
    ScoreCountMismatch {
        question: String,
        expected: usize,
        actual: usize,
    },
    #[error("question {question:?}: the model returned a score that is not a finite number")]
    NonFiniteScore { question: String },
    #[error(
        "question {question:?}: the model reported temperature {temperature}, which is not a positive number"
    )]
    UnusableTemperature { question: String, temperature: f64 },
    #[error(
        "question {question:?}: the model's {answer_kind} answer does not match the question's kind"
    )]
    AnswerKindMismatch {
        question: String,
        answer_kind: &'static str,
    },
    #[error(
        "question {question:?}: option {option_index} was served at probability {served} but its scores give {recomputed}; the scores and the answer disagree on option order, so calibration stopped"
    )]
    ScoresDisagreeWithAnswer {
        question: String,
        option_index: usize,
        served: f64,
        recomputed: f64,
    },
}

/// Checks `question_outcome` against `labelled_question` and keeps its scores.
pub fn scored_question_from_outcome(
    labelled_question: &LabelledQuestion,
    question_outcome: &QuestionOutcome,
    state_group_key: u64,
) -> Result<ScoredQuestion, ScoredQuestionError> {
    let question = &labelled_question.question;
    if question_outcome.question_name != question.name {
        return Err(ScoredQuestionError::AnsweredAnotherQuestion {
            asked: question.name.clone(),
            answered: question_outcome.question_name.clone(),
        });
    }
    check_scores_are_usable(labelled_question, question_outcome)?;
    let served_probabilities = served_probabilities(labelled_question, &question_outcome.answer)?;
    check_served_matches_scores(&question.name, question_outcome, &served_probabilities)?;
    Ok(ScoredQuestion {
        question_name: question.name.clone(),
        kind: question.kind(),
        option_logits: question_outcome.option_logits.clone(),
        shipped_temperature: question_outcome.applied_temperature,
        correct_option_index: labelled_question.correct_option_index,
        state_group_key,
    })
}

fn check_scores_are_usable(
    labelled_question: &LabelledQuestion,
    question_outcome: &QuestionOutcome,
) -> Result<(), ScoredQuestionError> {
    let question_name = &labelled_question.question.name;
    let expected_score_count = labelled_question.question.option_count();
    if question_outcome.option_logits.len() != expected_score_count {
        return Err(ScoredQuestionError::ScoreCountMismatch {
            question: question_name.clone(),
            expected: expected_score_count,
            actual: question_outcome.option_logits.len(),
        });
    }
    for option_logit in &question_outcome.option_logits {
        if !option_logit.is_finite() {
            return Err(ScoredQuestionError::NonFiniteScore {
                question: question_name.clone(),
            });
        }
    }
    let applied_temperature = question_outcome.applied_temperature;
    if !applied_temperature.is_finite() || applied_temperature <= 0.0 {
        return Err(ScoredQuestionError::UnusableTemperature {
            question: question_name.clone(),
            temperature: applied_temperature,
        });
    }
    Ok(())
}

/// The probabilities the person was served, in canonical option order.
fn served_probabilities(
    labelled_question: &LabelledQuestion,
    answer: &DecisionAnswer,
) -> Result<Vec<f64>, ScoredQuestionError> {
    let question = &labelled_question.question;
    let (answer_kind, probabilities) = match answer {
        DecisionAnswer::Choice(choice_answer) => {
            let mut probabilities = Vec::with_capacity(choice_answer.option_probabilities.len());
            for option_probability in &choice_answer.option_probabilities {
                probabilities.push(option_probability.probability);
            }
            (DecisionQuestionKind::Choice, probabilities)
        }
        DecisionAnswer::Score(score_answer) => (
            DecisionQuestionKind::Score,
            score_answer.level_probabilities.clone(),
        ),
        DecisionAnswer::Noul(noul_answer) => {
            let mut probabilities = vec![0.0; 2];
            probabilities[NOUL_FALSE_INDEX] = 1.0 - noul_answer.probability_true;
            probabilities[NOUL_TRUE_INDEX] = noul_answer.probability_true;
            (DecisionQuestionKind::Noul, probabilities)
        }
    };
    if answer_kind != question.kind() {
        return Err(ScoredQuestionError::AnswerKindMismatch {
            question: question.name.clone(),
            answer_kind: answer_kind.wire_name(),
        });
    }
    Ok(probabilities)
}

fn check_served_matches_scores(
    question_name: &str,
    question_outcome: &QuestionOutcome,
    served_probabilities: &[f64],
) -> Result<(), ScoredQuestionError> {
    let recomputed_probabilities = probabilities_at(
        &question_outcome.option_logits,
        question_outcome.applied_temperature,
    );
    for (option_index, recomputed) in recomputed_probabilities.iter().enumerate() {
        let served = served_probabilities
            .get(option_index)
            .copied()
            .unwrap_or(f64::NAN);
        // Written as "is close" so a missing (NaN) served probability, which
        // compares false with everything, counts as a mismatch.
        let is_close = (served - recomputed).abs() <= SERVED_PROBABILITY_TOLERANCE;
        if !is_close {
            return Err(ScoredQuestionError::ScoresDisagreeWithAnswer {
                question: question_name.to_owned(),
                option_index,
                served,
                recomputed: *recomputed,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "scored_question_tests.rs"]
mod tests;
