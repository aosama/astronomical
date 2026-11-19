//! Which of a job's questions a training uses, and what the worker knows
//! about each: its role, its target, its served scores, and where it came
//! from.
//!
//! A question whose options the model cannot tell apart (they ran out of
//! room) is left out and counted. A final-layer training also needs what
//! the final layers read for each question; a whole-model training runs the
//! model from the question itself, so it needs only where the question is.

use astronomical_decision_contract::decision_training_job::{DecisionTrainingJob, QuestionRole};
use astronomical_decision_contract::{DecisionFailure, HeadInputs, QuestionOutcome};
use astronomical_decision_training::trainable_head::TrainableHead;
use astronomical_decision_training::training_losses::tempered_probabilities;

use super::gpu_training::TrainingRow;

/// Where one training row's question is in the job.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowSource {
    pub example_index: usize,
    pub question_index: usize,
}

/// The questions training uses, and the host-side facts about each.
pub struct QuestionSelection<'read> {
    /// What the final layers read for each row; empty for a whole-model training.
    pub head_inputs: Vec<&'read HeadInputs>,
    pub row_sources: Vec<RowSource>,
    pub served_logits: Vec<Vec<f32>>,
    pub rows: Vec<TrainingRow>,
    pub training_rows: Vec<usize>,
    pub choosing_rows: Vec<usize>,
    pub check_rows: Vec<usize>,
    pub question_rows: Vec<Vec<Option<usize>>>,
    pub indistinguishable_question_count: usize,
}

/// The rows of `job`; `final_layers` is the head whose inputs each row must
/// carry, or `None` for a whole-model training.
pub fn select_questions<'read>(
    job: &DecisionTrainingJob,
    read_examples: &'read [Vec<QuestionOutcome>],
    final_layers: Option<&dyn TrainableHead>,
) -> Result<QuestionSelection<'read>, DecisionFailure> {
    let mut selection = QuestionSelection {
        head_inputs: Vec::new(),
        row_sources: Vec::new(),
        served_logits: Vec::new(),
        rows: Vec::new(),
        training_rows: Vec::new(),
        choosing_rows: Vec::new(),
        check_rows: Vec::new(),
        question_rows: Vec::with_capacity(read_examples.len()),
        indistinguishable_question_count: 0,
    };
    for (example_index, (example, question_outcomes)) in
        job.examples.iter().zip(read_examples).enumerate()
    {
        let mut example_rows = Vec::with_capacity(question_outcomes.len());
        for (question_index, (role, outcome)) in example
            .question_roles
            .iter()
            .zip(question_outcomes)
            .enumerate()
        {
            let row_source = RowSource {
                example_index,
                question_index,
            };
            example_rows.push(select_question(
                &mut selection,
                (*role, row_source),
                outcome,
                final_layers,
            )?);
        }
        selection.question_rows.push(example_rows);
    }
    Ok(selection)
}

/// Adds one question to the selection, or leaves it out when its options
/// read alike; returns its row.
fn select_question<'read>(
    selection: &mut QuestionSelection<'read>,
    (role, row_source): (QuestionRole, RowSource),
    outcome: &'read QuestionOutcome,
    final_layers: Option<&dyn TrainableHead>,
) -> Result<Option<usize>, DecisionFailure> {
    let option_count = outcome.option_logits.len();
    if (outcome.input_coverage.distinguishable_option_count as usize) < option_count {
        selection.indistinguishable_question_count += usize::from(role.correct_option().is_some());
        return Ok(None);
    }
    let row_head_inputs = match final_layers {
        Some(trainable_head) => Some(checked_head_inputs(outcome, trainable_head)?),
        None => None,
    };
    if let Some(correct_option) = role.correct_option()
        && correct_option >= option_count
    {
        return Err(DecisionFailure::invalid_request(&format!(
            "question {} has {option_count} options, so option {correct_option} cannot be its right answer",
            outcome.question_name
        )));
    }
    let row = selection.rows.len();
    let temperature = outcome.applied_temperature;
    let training_row = match role {
        QuestionRole::Check => TrainingRow::Check {
            shipped_probabilities: narrowed(&tempered_probabilities(
                &outcome.option_logits,
                temperature,
            )),
            temperature,
        },
        labelled_role => TrainingRow::Labelled {
            correct_option: labelled_role.correct_option().unwrap_or(0),
            temperature,
        },
    };
    match role {
        QuestionRole::Training { .. } => selection.training_rows.push(row),
        QuestionRole::Choosing { .. } => selection.choosing_rows.push(row),
        QuestionRole::Check => selection.check_rows.push(row),
        QuestionRole::Judging { .. } => {}
    }
    selection.rows.push(training_row);
    selection.row_sources.push(row_source);
    if let Some(head_inputs) = row_head_inputs {
        selection.head_inputs.push(head_inputs);
    }
    selection.served_logits.push(outcome.option_logits.clone());
    Ok(Some(row))
}

fn narrowed(probabilities: &[f64]) -> Vec<f32> {
    let mut narrowed_values = Vec::with_capacity(probabilities.len());
    for probability in probabilities {
        narrowed_values.push(*probability as f32);
    }
    narrowed_values
}

/// What the final layers read for `outcome`, checked against their shapes.
fn checked_head_inputs<'read>(
    outcome: &'read QuestionOutcome,
    trainable_head: &dyn TrainableHead,
) -> Result<&'read HeadInputs, DecisionFailure> {
    let Some(head_inputs) = &outcome.head_inputs else {
        return Err(DecisionFailure::fatal_execution(
            "the engine returned no head inputs",
        ));
    };
    if let Err(mismatch) = trainable_head.check_inputs(head_inputs) {
        return Err(DecisionFailure::fatal_execution(&mismatch.to_string()));
    }
    Ok(head_inputs)
}

#[cfg(test)]
#[path = "question_selection_tests.rs"]
mod tests;
