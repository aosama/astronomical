//! The training objective over a batch's questions, on the GPU.
//!
//! Every question is a row with a target distribution over its options, the
//! temperature it is served at, and a weight. A labelled question's target
//! is its label-smoothed answer; a check-set question's target is the
//! shipped model's own probabilities, so the same formula pulls training
//! toward the labels and toward keeping unlabelled answers. For both,
//! `d loss / d score = weight · (p − target) / T`, where `p` is the softmax
//! of the scores over `T`.

use mlx_bridge::MlxArray;

use super::gpu_math::{GpuMath, GpuResult};
use super::question_batch::QuestionBatch;

/// Each row's target, inverse temperature and weight, on the GPU.
pub(crate) struct RowTargets {
    /// `[questions, widest]`, zero on padding.
    pub(crate) targets: MlxArray,
    /// `[questions, 1]`.
    pub(crate) inverse_temperatures: MlxArray,
    /// `[questions, 1]`.
    pub(crate) row_weights: MlxArray,
}

/// One host row before upload.
pub(crate) struct HostRow<'row> {
    pub(crate) target: &'row [f32],
    pub(crate) temperature: f64,
    pub(crate) weight: f64,
}

impl RowTargets {
    pub(crate) fn upload(
        math: GpuMath<'_>,
        host_rows: &[HostRow<'_>],
        widest_option_count: usize,
    ) -> GpuResult<Self> {
        let mut targets = vec![0.0_f32; host_rows.len() * widest_option_count];
        let mut inverse_temperatures = Vec::with_capacity(host_rows.len());
        let mut row_weights = Vec::with_capacity(host_rows.len());
        for (row_index, host_row) in host_rows.iter().enumerate() {
            let row_start = row_index * widest_option_count;
            targets[row_start..row_start + host_row.target.len()].copy_from_slice(host_row.target);
            inverse_temperatures.push((1.0 / host_row.temperature) as f32);
            row_weights.push(host_row.weight as f32);
        }
        let row_count = host_rows.len();
        Ok(Self {
            targets: math.upload(&targets, &[row_count, widest_option_count])?,
            inverse_temperatures: math.upload(&inverse_temperatures, &[row_count, 1])?,
            row_weights: math.upload(&row_weights, &[row_count, 1])?,
        })
    }
}

/// `d loss / d score` for every option of the batch, in option order.
pub(crate) fn option_score_gradients(
    math: GpuMath<'_>,
    option_scores: &MlxArray,
    question_batch: &QuestionBatch,
    row_targets: &RowTargets,
) -> GpuResult<MlxArray> {
    let runtime = math.runtime;
    let tempered = runtime.multiply(
        &question_batch.padded(math, option_scores)?,
        &row_targets.inverse_temperatures,
    )?;
    let probabilities = runtime.softmax_axis(&tempered, 1)?;
    let row_scale =
        runtime.multiply(&row_targets.row_weights, &row_targets.inverse_temperatures)?;
    let padded_gradients = runtime.multiply(
        &runtime.subtract(&probabilities, &row_targets.targets)?,
        &row_scale,
    )?;
    question_batch.unpadded(math, &padded_gradients)
}

/// The mean of `−Σ target · log p` over the batch's rows: with one-hot
/// targets, the mean surprise at the right answer.
pub(crate) fn mean_cross_entropy(
    math: GpuMath<'_>,
    option_scores: &MlxArray,
    question_batch: &QuestionBatch,
    row_targets: &RowTargets,
) -> GpuResult<f64> {
    let runtime = math.runtime;
    let tempered = runtime.multiply(
        &question_batch.padded(math, option_scores)?,
        &row_targets.inverse_temperatures,
    )?;
    let log_probabilities =
        runtime.subtract(&tempered, &runtime.logsumexp_axis(&tempered, 1, true)?)?;
    let row_losses = runtime.negative(&runtime.sum_axis(
        &runtime.multiply(&row_targets.targets, &log_probabilities)?,
        1,
        false,
    )?)?;
    let mean_loss = math.download(&runtime.mean(&row_losses, false)?)?;
    Ok(f64::from(mean_loss.first().copied().unwrap_or(f32::NAN)))
}

#[cfg(test)]
#[path = "row_objective_tests.rs"]
mod tests;
