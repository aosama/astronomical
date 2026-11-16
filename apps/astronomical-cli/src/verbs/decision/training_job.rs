//! Building the job the worker trains on, and reading its outcome back as
//! the scores the gate judges.
//!
//! The split is decided here, not in the worker: each labelled question
//! goes in with its part (training, choosing or judging) and right answer,
//! and the check-set questions go in unlabelled. What comes back is every
//! question's served answer beside the trained head's scores, so the gate,
//! the report and the written-folder check all run on this side, on
//! numbers the worker produced once.

use std::collections::HashMap;

use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, QuestionRole, TrainedQuestion, TrainingExample,
    TrainingScope,
};
use astronomical_decision_contract::{QuestionOutcome, parse_system_one_questions};
use astronomical_decision_training::holdout_split::state_group_key;
use astronomical_decision_training::labelled_example::LabelledExample;
use astronomical_decision_training::training_losses::tempered_probabilities;
use astronomical_decision_training::training_split::{SplitPart, split_parts_by_state};
use astronomical_decision_training::training_verdict::{CheckPair, ScoredPair, TrainingScores};

use super::calibrate::scored_example;
use super::check_set::CheckExample;
use super::decision_tool::DecisionToolError;

/// How many labelled questions each part holds, for the report.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct PartCounts {
    pub(super) training: usize,
    pub(super) choosing: usize,
    pub(super) judging: usize,
}

/// The job, and which part each labelled example is in.
pub(super) struct PreparedJob {
    pub(super) job: DecisionTrainingJob,
    /// One per labelled example, in example order.
    pub(super) example_parts: Vec<SplitPart>,
    pub(super) part_counts: PartCounts,
}

impl PreparedJob {
    /// Every example the job sends: the labelled ones, then the check set.
    pub(super) fn sent_example_count(&self) -> usize {
        self.job.examples.len()
    }
}

/// The labelled examples, each question with its part and right answer,
/// followed by the check set with every question marked as a check.
pub(super) fn training_job(
    labelled_examples: &[LabelledExample],
    check_examples: &[CheckExample],
    scope: TrainingScope,
) -> Result<PreparedJob, DecisionToolError> {
    let mut state_keys = Vec::with_capacity(labelled_examples.len());
    for labelled_example in labelled_examples {
        state_keys.push(state_group_key(&labelled_example.state));
    }
    let part_by_state = split_parts_by_state(&state_keys);
    let mut prepared = PreparedJob {
        job: DecisionTrainingJob {
            examples: Vec::with_capacity(labelled_examples.len() + check_examples.len()),
            scope,
        },
        example_parts: Vec::with_capacity(labelled_examples.len()),
        part_counts: PartCounts::default(),
    };
    for (labelled_example, state_key) in labelled_examples.iter().zip(&state_keys) {
        let part = part_of(&part_by_state, *state_key);
        prepared.example_parts.push(part);
        prepared.job.examples.push(labelled_training_example(
            labelled_example,
            part,
            &mut prepared.part_counts,
        ));
    }
    for check_example in check_examples {
        prepared
            .job
            .examples
            .push(check_training_example(check_example)?);
    }
    Ok(prepared)
}

/// Every key came from the same examples the parts were split from, so a
/// missing one cannot happen; training is the part that judges nothing.
fn part_of(part_by_state: &HashMap<u64, SplitPart>, state_key: u64) -> SplitPart {
    match part_by_state.get(&state_key) {
        Some(part) => *part,
        None => SplitPart::Training,
    }
}

fn labelled_training_example(
    labelled_example: &LabelledExample,
    part: SplitPart,
    part_counts: &mut PartCounts,
) -> TrainingExample {
    let mut question_roles = Vec::with_capacity(labelled_example.labelled_questions.len());
    for labelled_question in &labelled_example.labelled_questions {
        let correct_option = labelled_question.correct_option_index;
        question_roles.push(match part {
            SplitPart::Training => QuestionRole::Training { correct_option },
            SplitPart::Choosing => QuestionRole::Choosing { correct_option },
            SplitPart::Judging => QuestionRole::Judging { correct_option },
        });
    }
    let question_count = question_roles.len();
    match part {
        SplitPart::Training => part_counts.training += question_count,
        SplitPart::Choosing => part_counts.choosing += question_count,
        SplitPart::Judging => part_counts.judging += question_count,
    }
    TrainingExample {
        state: labelled_example.state.clone(),
        questions: labelled_example.questions_document.clone(),
        question_roles,
    }
}

fn check_training_example(
    check_example: &CheckExample,
) -> Result<TrainingExample, DecisionToolError> {
    let check_questions = match parse_system_one_questions(&check_example.questions_document) {
        Ok(check_questions) => check_questions,
        Err(definition_error) => {
            return Err(DecisionToolError::CheckSetUnreadable(
                definition_error.to_string(),
            ));
        }
    };
    Ok(TrainingExample {
        state: check_example.state.clone(),
        questions: check_example.questions_document.clone(),
        question_roles: vec![QuestionRole::Check; check_questions.len()],
    })
}

/// The choosing, judging and check scores the gate reads. Each labelled
/// answer passes calibration's own checks first; questions the worker left
/// out (their options read alike) carry no trained scores and are skipped.
pub(super) fn training_scores(
    labelled_examples: &[LabelledExample],
    example_parts: &[SplitPart],
    sent_example_count: usize,
    outcome: &DecisionTrainingOutcome,
) -> Result<TrainingScores, DecisionToolError> {
    let labelled_count = labelled_examples.len();
    if outcome.examples.len() != sent_example_count
        || sent_example_count < labelled_count
        || example_parts.len() != labelled_count
    {
        return Err(DecisionToolError::TrainingFailed(format!(
            "the worker answered {} examples but {sent_example_count} were sent",
            outcome.examples.len()
        )));
    }
    let mut training_scores = TrainingScores {
        choosing: Vec::new(),
        judging: Vec::new(),
        check: Vec::new(),
    };
    for ((labelled_example, part), trained_questions) in labelled_examples
        .iter()
        .zip(example_parts)
        .zip(&outcome.examples)
    {
        let scored_pairs = labelled_pairs(labelled_example, trained_questions)?;
        match part {
            SplitPart::Training => {}
            SplitPart::Choosing => training_scores.choosing.extend(scored_pairs),
            SplitPart::Judging => training_scores.judging.extend(scored_pairs),
        }
    }
    for trained_questions in &outcome.examples[labelled_count..] {
        training_scores
            .check
            .extend(check_pairs(trained_questions)?);
    }
    Ok(training_scores)
}

fn labelled_pairs(
    labelled_example: &LabelledExample,
    trained_questions: &[TrainedQuestion],
) -> Result<Vec<ScoredPair>, DecisionToolError> {
    let mut served_outcomes: Vec<QuestionOutcome> = Vec::with_capacity(trained_questions.len());
    for trained_question in trained_questions {
        served_outcomes.push(trained_question.served.clone());
    }
    let scored_questions = scored_example(labelled_example, &served_outcomes)?;
    let mut scored_pairs = Vec::with_capacity(scored_questions.len());
    for (scored_question, trained_question) in scored_questions.into_iter().zip(trained_questions) {
        if trained_question.trained_logits.is_empty() {
            continue;
        }
        check_score_count(trained_question)?;
        scored_pairs.push(ScoredPair {
            served: scored_question,
            trained_logits: trained_question.trained_logits.clone(),
        });
    }
    Ok(scored_pairs)
}

fn check_pairs(trained_questions: &[TrainedQuestion]) -> Result<Vec<CheckPair>, DecisionToolError> {
    let mut check_pairs = Vec::with_capacity(trained_questions.len());
    for trained_question in trained_questions {
        if trained_question.trained_logits.is_empty() {
            continue;
        }
        check_score_count(trained_question)?;
        let served = &trained_question.served;
        check_pairs.push(CheckPair {
            served_probabilities: tempered_probabilities(
                &served.option_logits,
                served.applied_temperature,
            ),
            trained_logits: trained_question.trained_logits.clone(),
        });
    }
    Ok(check_pairs)
}

/// The trained head scores the same options the model served.
fn check_score_count(trained_question: &TrainedQuestion) -> Result<(), DecisionToolError> {
    let served = &trained_question.served;
    if trained_question.trained_logits.len() == served.option_logits.len() {
        return Ok(());
    }
    Err(DecisionToolError::TrainingFailed(format!(
        "question {} came back with {} trained scores for {} options",
        served.question_name,
        trained_question.trained_logits.len(),
        served.option_logits.len()
    )))
}

#[cfg(test)]
#[path = "training_job_tests.rs"]
mod tests;
