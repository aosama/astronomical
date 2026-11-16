//! How serving grouped each example's questions into forward passes, so a
//! whole-model training can score its questions exactly as serving does.
//!
//! Serving runs all of a request's questions together in one pass. What
//! else is in a pass changes a question's score: padding changes which
//! kernels attention takes, and more rows change how MLX multiplies the
//! 8-bit weights (on this hardware larger products run in TF32). Across
//! Laya's thirty layers that grows to as much as 0.02 in a logit, measured
//! on the real model: a question scored beside others, even others of its
//! own length, scores a little differently from the same question alone.
//! Training's checks compare its scores with serving's to well under that,
//! so wherever a score is compared (the parity check before training, the
//! trained scores the gate and the saved folder are judged by) the question
//! runs in the very pass serving ran it in, beside the same questions and
//! nothing else. Gradient steps need no such care and use larger batches.

use astronomical_decision_contract::decision_training_job::TrainingExample;
use astronomical_decision_contract::laya_rendering::laya_question_type_index;
use astronomical_decision_contract::{DecisionFailure, parse_system_one_questions};

use crate::decision_models::laya::batching::{PADDED_TOKENS_PER_PASS, plan_forward_passes};
use crate::decision_models::laya::loaded_state::LoadedLayaState;
use crate::decision_models::training::question_selection::RowSource;

/// One question as the model reads it.
#[derive(Clone)]
pub(crate) struct SequenceRow {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) question_type_index: u32,
    pub(crate) option_marker_positions: Vec<usize>,
}

/// One forward pass serving ran: its questions, and which training row each is.
pub(crate) struct ServedPass {
    pub(crate) sequence_rows: Vec<SequenceRow>,
    /// Each question's training row, or `None` for a question training left out.
    pub(crate) training_rows: Vec<Option<usize>>,
}

/// Every pass serving ran over a training's examples.
pub(crate) struct ServedBatches {
    pub(crate) passes: Vec<ServedPass>,
    /// For each training row, its pass.
    row_passes: Vec<usize>,
}

impl ServedBatches {
    /// Lays out every example as serving did, and returns each training
    /// row's own question beside the passes. `row_sources` come in example
    /// order, one per training row.
    pub(crate) fn build(
        loaded_state: &LoadedLayaState,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
    ) -> Result<(Self, Vec<SequenceRow>), DecisionFailure> {
        let mut served_batches = Self {
            passes: Vec::new(),
            row_passes: vec![0; row_sources.len()],
        };
        let mut training_sequence_rows = Vec::with_capacity(row_sources.len());
        let mut next_source = 0;
        while next_source < row_sources.len() {
            let example_index = row_sources[next_source].example_index;
            let mut example_end = next_source;
            while example_end < row_sources.len()
                && row_sources[example_end].example_index == example_index
            {
                example_end += 1;
            }
            served_batches.add_example(
                loaded_state,
                &examples[example_index],
                (next_source, &row_sources[next_source..example_end]),
                &mut training_sequence_rows,
            )?;
            next_source = example_end;
        }
        Ok((served_batches, training_sequence_rows))
    }

    /// Adds one example's passes; `first_row` is its first training row.
    fn add_example(
        &mut self,
        loaded_state: &LoadedLayaState,
        example: &TrainingExample,
        (first_row, example_sources): (usize, &[RowSource]),
        training_sequence_rows: &mut Vec<SequenceRow>,
    ) -> Result<(), DecisionFailure> {
        let parsed_questions = match parse_system_one_questions(&example.questions) {
            Ok(parsed_questions) => parsed_questions,
            Err(question_error) => {
                return Err(DecisionFailure::invalid_request(
                    &question_error.to_string(),
                ));
            }
        };
        let question_sequences =
            loaded_state.build_question_sequences(&example.state, &parsed_questions)?;
        let mut question_rows: Vec<Option<usize>> = vec![None; question_sequences.len()];
        for (source_offset, row_source) in example_sources.iter().enumerate() {
            question_rows[row_source.question_index] = Some(first_row + source_offset);
        }
        let mut sequence_lengths = Vec::with_capacity(question_sequences.len());
        let mut example_rows = Vec::with_capacity(question_sequences.len());
        for (question_sequence, question) in question_sequences.iter().zip(&parsed_questions) {
            sequence_lengths.push(question_sequence.token_ids.len());
            example_rows.push(SequenceRow {
                token_ids: question_sequence.token_ids.clone(),
                question_type_index: laya_question_type_index(question.kind()),
                option_marker_positions: question_sequence.option_marker_positions.clone(),
            });
        }
        for row_source in example_sources {
            training_sequence_rows.push(example_rows[row_source.question_index].clone());
        }
        // The same planning serving's `decide` runs over the same questions.
        for pass_questions in plan_forward_passes(&sequence_lengths, PADDED_TOKENS_PER_PASS) {
            let pass_index = self.passes.len();
            let mut served_pass = ServedPass {
                sequence_rows: Vec::with_capacity(pass_questions.len()),
                training_rows: Vec::with_capacity(pass_questions.len()),
            };
            for question_index in pass_questions {
                served_pass
                    .sequence_rows
                    .push(example_rows[question_index].clone());
                served_pass
                    .training_rows
                    .push(question_rows[question_index]);
                if let Some(training_row) = question_rows[question_index] {
                    self.row_passes[training_row] = pass_index;
                }
            }
            self.passes.push(served_pass);
        }
        Ok(())
    }

    /// The passes that hold `training_rows`, each to run on its own.
    pub(crate) fn passes_holding(&self, training_rows: &[usize]) -> Vec<usize> {
        let mut needed_passes: Vec<usize> = Vec::with_capacity(training_rows.len());
        for training_row in training_rows {
            needed_passes.push(self.row_passes[*training_row]);
        }
        needed_passes.sort_unstable();
        needed_passes.dedup();
        needed_passes
    }
}

#[cfg(test)]
#[path = "served_batches_tests.rs"]
mod tests;
