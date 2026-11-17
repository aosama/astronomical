//! What the training loop trains, kept apart from the loop itself.
//!
//! The loop (runs, shuffled batches with check-set questions mixed in,
//! anchored Adam, the choosing score, early stopping) is the same whether a
//! training moves only the final layers or add-on weights through the whole
//! model. What differs is how a batch of questions is scored from the
//! parameters and how the objective's gradient is found: the final layers
//! score inputs the model read once and have hand-written gradients, while
//! a whole-model training runs the model from its tokens and asks MLX for
//! the gradient.

use mlx_bridge::MlxArray;

use super::gpu_heads::{GpuHeadFamily, GpuHeadInputs};
use super::gpu_math::{GpuMath, GpuResult};
use super::question_batch::QuestionBatch;
use super::row_objective::{RowTargets, option_score_gradients};

/// Scores questions from trainable parameters.
pub(crate) trait TrainableScorer {
    /// The batch's option scores under `parameters`, in option order.
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray>;

    /// The gradient, one array per parameter, of
    /// `Σ weight · −Σ target · log softmax(score / T)` over the batch's rows.
    fn parameter_gradients(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
        row_targets: &RowTargets,
    ) -> GpuResult<Vec<MlxArray>>;

    /// How many different texts so far were too long for a pass's memory
    /// budget and had every layer recomputed during the backward pass; only
    /// a whole-model training ever does.
    fn recomputed_text_count(&self) -> usize {
        0
    }
}

/// A family's final layers, scoring what the rest of the model read once.
pub(crate) struct FinalLayersScorer {
    pub(crate) family: GpuHeadFamily,
    pub(crate) head_inputs: GpuHeadInputs,
}

impl TrainableScorer for FinalLayersScorer {
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        self.family
            .option_scores(math, parameters, &self.head_inputs, question_batch)
    }

    fn parameter_gradients(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
        row_targets: &RowTargets,
    ) -> GpuResult<Vec<MlxArray>> {
        let option_scores = self.option_scores(math, parameters, question_batch)?;
        let score_gradients =
            option_score_gradients(math, &option_scores, question_batch, row_targets)?;
        self.family.parameter_gradients(
            math,
            parameters,
            &self.head_inputs,
            question_batch,
            &score_gradients,
        )
    }
}
