//! Questions with different option counts, laid out for batched GPU passes.
//!
//! A family's trainable part scores options, one row per option, all
//! questions' options stacked. The loss needs a softmax over each
//! question's own options, so the option scores are gathered into a
//! `[questions, widest option count]` matrix, padding filled from one extra
//! slot that holds a large negative score: its probability is exactly zero
//! and, unlike negative infinity, it never makes `0 · ∞` in a gradient.

use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult};

/// The score a padding slot gets; far below any real score.
pub const PADDING_SCORE: f32 = -1.0e30;

/// Where every question's options sit among all the dataset's options.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionLayout {
    option_offsets: Vec<usize>,
    option_counts: Vec<usize>,
}

impl OptionLayout {
    pub fn new(option_counts: Vec<usize>) -> Self {
        let mut option_offsets = Vec::with_capacity(option_counts.len());
        let mut next_offset = 0;
        for option_count in &option_counts {
            option_offsets.push(next_offset);
            next_offset += option_count;
        }
        Self {
            option_offsets,
            option_counts,
        }
    }

    pub fn option_count(&self, question_row: usize) -> usize {
        self.option_counts[question_row]
    }

    /// Host-side bookkeeping for the questions `question_rows`, in that order.
    pub fn batch_rows(&self, question_rows: &[usize]) -> BatchRows {
        let mut batch_rows = BatchRows::default();
        for (batch_question, question_row) in question_rows.iter().enumerate() {
            let option_count = self.option_counts[*question_row];
            batch_rows.widest_option_count = batch_rows.widest_option_count.max(option_count);
            for option_index in 0..option_count {
                batch_rows
                    .dataset_option_rows
                    .push(self.option_offsets[*question_row] + option_index);
                batch_rows.option_question.push(batch_question);
                batch_rows.option_index.push(option_index);
            }
        }
        batch_rows.dataset_question_rows = question_rows.to_vec();
        batch_rows
    }
}

/// A batch's indices before they go to the GPU.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchRows {
    pub dataset_question_rows: Vec<usize>,
    pub dataset_option_rows: Vec<usize>,
    pub option_question: Vec<usize>,
    pub option_index: Vec<usize>,
    pub widest_option_count: usize,
}

/// A batch's indices on the GPU.
pub struct QuestionBatch {
    pub question_count: usize,
    pub widest_option_count: usize,
    /// Each batch question's row among the dataset's questions.
    pub dataset_question_rows: MlxArray,
    /// The same rows on the host, for scorers that run the model per question.
    pub dataset_question_row_numbers: Vec<usize>,
    /// Each batch option's row among the dataset's options.
    pub dataset_option_rows: MlxArray,
    /// `[options, questions]`, one where an option belongs to a question:
    /// `assignment · per_question` gives each option its question's row,
    /// and `assignmentᵀ · per_option` sums options back into questions.
    pub option_assignment: MlxArray,
    /// `[questions · widest]` positions into the option scores with one
    /// padding slot appended at the end.
    padded_option_positions: MlxArray,
    /// Each option's place in the flattened `[questions, widest]` matrix.
    flat_option_positions: MlxArray,
}

impl QuestionBatch {
    pub fn upload(math: GpuMath<'_>, batch_rows: &BatchRows) -> GpuResult<Self> {
        let question_count = batch_rows.dataset_question_rows.len();
        let widest = batch_rows.widest_option_count.max(1);
        let option_total = batch_rows.dataset_option_rows.len();
        let mut padded_positions = vec![option_total; question_count * widest];
        let mut assignment = vec![0.0_f32; option_total * question_count];
        let mut flat_positions = Vec::with_capacity(option_total);
        for (batch_option, (question, option_index)) in batch_rows
            .option_question
            .iter()
            .zip(&batch_rows.option_index)
            .enumerate()
        {
            let flat_position = question * widest + option_index;
            padded_positions[flat_position] = batch_option;
            flat_positions.push(flat_position);
            assignment[batch_option * question_count + question] = 1.0;
        }
        Ok(Self {
            question_count,
            widest_option_count: widest,
            dataset_question_rows: math.upload_indices(&batch_rows.dataset_question_rows)?,
            dataset_question_row_numbers: batch_rows.dataset_question_rows.clone(),
            dataset_option_rows: math.upload_indices(&batch_rows.dataset_option_rows)?,
            option_assignment: math.upload(&assignment, &[option_total, question_count])?,
            padded_option_positions: math.upload_indices(&padded_positions)?,
            flat_option_positions: math.upload_indices(&flat_positions)?,
        })
    }

    /// `[questions, widest]` scores, padding at a large negative score.
    pub fn padded(&self, math: GpuMath<'_>, option_scores: &MlxArray) -> GpuResult<MlxArray> {
        let runtime = math.runtime;
        let padding = runtime.array_from_f32(&[PADDING_SCORE], &[1])?;
        let extended = runtime.concatenate_axis(&[option_scores, &padding], 0)?;
        let gathered = runtime.take_axis(&extended, &self.padded_option_positions, 0)?;
        Ok(runtime.reshape(
            &gathered,
            &[self.question_count as i32, self.widest_option_count as i32],
        )?)
    }

    /// Each option's value from a `[questions, widest]` matrix, back in option order.
    pub fn unpadded(&self, math: GpuMath<'_>, padded_values: &MlxArray) -> GpuResult<MlxArray> {
        let runtime = math.runtime;
        let flat = runtime.reshape(
            padded_values,
            &[(self.question_count * self.widest_option_count) as i32],
        )?;
        Ok(runtime.take_axis(&flat, &self.flat_option_positions, 0)?)
    }
}

#[cfg(test)]
#[path = "question_batch_tests.rs"]
mod tests;
