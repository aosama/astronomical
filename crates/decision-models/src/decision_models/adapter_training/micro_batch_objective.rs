//! The training objective over a few questions run through the whole model
//! at once, built as an MLX graph so MLX can differentiate it.
//!
//! It is the same objective final-layer training uses
//! (`training::row_objective`): for each question,
//! `weight · −Σ target · log softmax(score / T)`. There the gradient is
//! written by hand; here MLX traces it back through every layer.
//!
//! A batch of questions is too large to run backwards through the whole
//! model at once: every layer keeps its activations until the backward pass,
//! so memory grows with the padded tokens in flight. It is therefore split
//! into micro-batches of bounded size, each differentiated on its own and
//! evaluated before the next starts, and their gradients summed. Each
//! question carries its share of the batch's weight already, so the sum is
//! exactly the whole batch's gradient.

use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};

use crate::decision_models::training::question_batch::PADDING_SCORE;
use crate::decision_models::training::row_objective::RowTargets;

/// One micro-batch's rows of the batch's targets, `widest` the batch's
/// widest option count.
pub struct MicroBatchTargets {
    /// `[rows, widest]`, zero on padding.
    targets: MlxArray,
    /// `[rows, 1]`.
    inverse_temperatures: MlxArray,
    /// `[rows, 1]`.
    row_weights: MlxArray,
    /// `[rows, widest]`: zero on a row's own options, a large negative
    /// number past them, so padding never takes probability.
    option_mask: MlxArray,
    widest_option_count: usize,
}

impl MicroBatchTargets {
    /// The rows at `batch_positions` of `row_targets`, each with
    /// `option_counts` options.
    pub fn take(
        context: &MlxBindingsContext,
        row_targets: &RowTargets,
        batch_positions: &[usize],
        option_counts: &[usize],
    ) -> Result<Self, MlxCError> {
        let widest_option_count = row_targets.targets.shape()[1] as usize;
        let mut position_values = Vec::with_capacity(batch_positions.len());
        for batch_position in batch_positions {
            // A batch holds a few dozen questions.
            position_values.push(*batch_position as i32);
        }
        let positions =
            context.array_from_i32(&position_values, &[position_values.len() as i32])?;
        let mut mask_values = vec![0.0_f32; batch_positions.len() * widest_option_count];
        for (row_index, option_count) in option_counts.iter().enumerate() {
            let row_start = row_index * widest_option_count;
            for padded_value in
                &mut mask_values[row_start + option_count..row_start + widest_option_count]
            {
                *padded_value = PADDING_SCORE;
            }
        }
        Ok(Self {
            targets: context.take_axis(&row_targets.targets, &positions, 0)?,
            inverse_temperatures: context.take_axis(
                &row_targets.inverse_temperatures,
                &positions,
                0,
            )?,
            row_weights: context.take_axis(&row_targets.row_weights, &positions, 0)?,
            option_mask: context.array_from_f32(
                &mask_values,
                &[batch_positions.len() as i32, widest_option_count as i32],
            )?,
            widest_option_count,
        })
    }

    /// The batch's widest option count: every row's scores are padded to it.
    pub fn widest_option_count(&self) -> usize {
        self.widest_option_count
    }

    /// `Σ weight · −Σ target · log softmax(score / T)` over the rows, a scalar,
    /// from `[rows, options]` logits whose padding slots hold any value.
    pub fn weighted_cross_entropy(
        &self,
        context: &MlxBindingsContext,
        padded_logits: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let logits_shape = padded_logits.shape();
        let missing_columns = self.widest_option_count as i32 - logits_shape[1];
        let full_width_logits = if missing_columns > 0 {
            let filler =
                context.zeros(&[logits_shape[0], missing_columns], padded_logits.dtype())?;
            context.concatenate_axis(&[padded_logits, &filler], 1)?
        } else {
            padded_logits.retain()?
        };
        let masked = context.add(&full_width_logits, &self.option_mask)?;
        let tempered = context.multiply(&masked, &self.inverse_temperatures)?;
        let log_probabilities =
            context.subtract(&tempered, &context.logsumexp_axis(&tempered, 1, true)?)?;
        let row_surprises = context.negative(&context.sum_axis(
            &context.multiply(&self.targets, &log_probabilities)?,
            1,
            true,
        )?)?;
        context.sum(&context.multiply(&row_surprises, &self.row_weights)?, false)
    }
}

/// Adds `micro_batch_gradients` into `summed_gradients` and evaluates the
/// sums, so the micro-batch's activations are freed before the next runs.
pub fn add_and_evaluate(
    context: &MlxBindingsContext,
    summed_gradients: &mut Vec<MlxArray>,
    micro_batch_gradients: Vec<MlxArray>,
) -> Result<(), MlxCError> {
    if summed_gradients.is_empty() {
        *summed_gradients = micro_batch_gradients;
    } else {
        for (summed_gradient, micro_batch_gradient) in
            summed_gradients.iter_mut().zip(&micro_batch_gradients)
        {
            *summed_gradient = context.add(summed_gradient, micro_batch_gradient)?;
        }
    }
    let mut gradient_references = Vec::with_capacity(summed_gradients.len());
    for summed_gradient in summed_gradients.iter() {
        gradient_references.push(summed_gradient);
    }
    context.evaluate_arrays(&gradient_references)
}

#[cfg(test)]
#[path = "micro_batch_objective_tests.rs"]
mod tests;
