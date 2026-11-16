//! Scoring and differentiating Laya questions through the whole model, with
//! add-on weights in every layer and the final scorer trainable.
//!
//! Every question runs from its tokens through the same forward pass that
//! serves it ([`LoadedLayaState::padded_option_logits`]), with the trained
//! parts read from the parameters being moved. Scores that are compared
//! with serving's run each question in the pass serving ran it in, so what
//! training scores is what the trained folder will serve; gradient steps
//! use larger padded batches, where rounding that small does not matter.
//!
//! Every pass leaves freed GPU buffers in MLX's buffer cache, and passes of
//! different shapes leave buffers nothing reuses. Serving empties the cache
//! after each request; a training runs thousands of passes inside one
//! request, and left alone the cache grew until the Mac swapped (found
//! live: steps slowed from 6 to 40 seconds and the desktop froze). So the
//! cache is reclaimed after every pass once it holds more than the
//! runtime's reclaim threshold.

use std::collections::BTreeMap;
use std::rc::Rc;

use astronomical_decision_contract::DecisionFailure;
use astronomical_decision_contract::decision_training_job::TrainingExample;
use astronomical_decision_training::head_parameters::ParameterLayout;
use astronomical_runtime_integration::MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES;
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};

use crate::decision_models::adapter_training::loss_gradient::{TrainableLoss, loss_gradients};
use crate::decision_models::adapter_training::micro_batch_objective::{
    MicroBatchTargets, add_and_evaluate,
};
use crate::decision_models::adapter_training::whole_model_layout::{
    ADAPTER_RANK, WholeModelLayout,
};
use crate::decision_models::laya::adapter_targets::laya_adapter_targets;
use crate::decision_models::laya::batching::plan_forward_passes;
use crate::decision_models::laya::loaded_state::{ForwardRows, LoadedLayaState, TrainedParts};
use crate::decision_models::laya::served_batches::{SequenceRow, ServedBatches};
use crate::decision_models::training::gpu_math::{GpuMath, GpuResult, GpuTrainingError};
use crate::decision_models::training::question_batch::QuestionBatch;
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::row_objective::RowTargets;
use crate::decision_models::training::trainable_scorer::TrainableScorer;
use crate::decision_models::training::training_session::WholeModelScorer;

/// What one padded token costs while a micro-batch runs backwards through
/// the model: every layer keeps its activations for the backward pass,
/// measured on Laya at 2,048 tokens peaking at 7.9 GB (about 3.6 MB per
/// token after the model itself).
const TRAINING_BYTES_PER_PADDED_TOKEN: usize = 3_800_000;
/// The share of the runtime's memory limit one micro-batch may use, so the
/// model, the optimizer and macOS keep the rest.
const TRAINING_MEMORY_LIMIT_DIVISOR: usize = 4;
/// The largest micro-batch, in padded tokens: about 4 GB, where the
/// measured step time stops improving with size.
const MAXIMUM_TRAINING_PADDED_TOKENS_PER_PASS: usize = 1_024;
/// The smallest: one long question still forms a pass of its own.
const MINIMUM_TRAINING_PADDED_TOKENS_PER_PASS: usize = 128;

/// The padded tokens one micro-batch may hold under a runtime memory limit
/// of `memory_limit_bytes`: a quarter of it, never more than the measured
/// sweet spot. A Mac with less memory trains in smaller batches, so it
/// trains more slowly rather than swapping; found live, when fixed
/// 2,048-token batches froze a 48 GB Mac.
pub(crate) fn training_tokens_per_pass(memory_limit_bytes: usize) -> usize {
    (memory_limit_bytes / TRAINING_MEMORY_LIMIT_DIVISOR / TRAINING_BYTES_PER_PADDED_TOKEN).clamp(
        MINIMUM_TRAINING_PADDED_TOKENS_PER_PASS,
        MAXIMUM_TRAINING_PADDED_TOKENS_PER_PASS,
    )
}

/// Laya's whole model, as the training loop's scorer.
pub(crate) struct LayaWholeModelScorer {
    pub(crate) loaded_state: Rc<LoadedLayaState>,
    /// Every training row's question, in row order.
    pub(crate) sequence_rows: Rc<Vec<SequenceRow>>,
    /// How serving grouped the questions, for scores that are compared.
    pub(crate) served_batches: Rc<ServedBatches>,
    pub(crate) layout: Rc<WholeModelLayout>,
    /// The padded tokens one gradient micro-batch may hold.
    pub(crate) tokens_per_pass: usize,
}

impl LayaWholeModelScorer {
    /// The whole model over the questions at `row_sources`, with an add-on
    /// on every 8-bit layer and the final layers laid out as `scorer_layout`.
    pub(crate) fn for_rows(
        loaded_state: &Rc<LoadedLayaState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        if !loaded_state.adapters.is_empty() {
            // Training on top of earlier add-ons would need them as the
            // starting point and as what the anchor pulls toward; until
            // then, refusing is the honest answer.
            return Err(DecisionFailure::invalid_request(
                "this model was already trained as a whole; train the original model instead",
            ));
        }
        let adapter_targets = laya_adapter_targets(
            &loaded_state.tensors,
            &loaded_state.configuration,
            &loaded_state.weight_layout,
        )?;
        let layout = Rc::new(WholeModelLayout {
            scorer_layout: scorer_layout.clone(),
            adapter_targets,
            adapter_rank: ADAPTER_RANK,
        });
        let (served_batches, sequence_rows) =
            ServedBatches::build(loaded_state, examples, row_sources)?;
        let memory_limit_bytes = match loaded_state.runtime.configured_memory_limit_bytes() {
            Ok(memory_limit_bytes) => memory_limit_bytes,
            Err(limit_error) => {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "the training's memory limit could not be read: {limit_error}"
                )));
            }
        };
        let scorer = Self {
            tokens_per_pass: training_tokens_per_pass(memory_limit_bytes),
            loaded_state: Rc::clone(loaded_state),
            sequence_rows: Rc::new(sequence_rows),
            served_batches: Rc::new(served_batches),
            layout: Rc::clone(&layout),
        };
        Ok(WholeModelScorer {
            scorer: Box::new(scorer),
            layout,
        })
    }

    /// The batch's positions grouped into micro-batches by padded size.
    fn micro_batches(&self, row_numbers: &[usize]) -> Vec<Vec<usize>> {
        let mut sequence_lengths = Vec::with_capacity(row_numbers.len());
        for row_number in row_numbers {
            sequence_lengths.push(self.sequence_rows[*row_number].token_ids.len());
        }
        plan_forward_passes(&sequence_lengths, self.tokens_per_pass)
    }
}

impl TrainableScorer for LayaWholeModelScorer {
    /// Scores each question in the pass serving ran it in, so these scores
    /// compare with serving's to rounding; see [`ServedBatches`].
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        let row_numbers = &question_batch.dataset_question_row_numbers;
        let trained_parts = self.layout.trained_parts(parameters)?;
        let parts = TrainedParts {
            tuned_tensors: &trained_parts.tuned_tensors,
            adapters: Some(&trained_parts.adapters),
        };
        let mut scores_by_row: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for pass_index in self.served_batches.passes_holding(row_numbers) {
            let served_pass = &self.served_batches.passes[pass_index];
            let group_rows: Vec<&SequenceRow> = served_pass.sequence_rows.iter().collect();
            let group_training_rows = &served_pass.training_rows;
            let padded_logits = self
                .loaded_state
                .padded_option_logits(&forward_rows(&group_rows), &parts)
                .map_err(scoring_failure)?;
            let flat_logits = math.download(&padded_logits)?;
            let widest_option_count = flat_logits.len() / group_rows.len().max(1);
            for (group_position, (sequence_row, training_row)) in
                group_rows.iter().zip(group_training_rows).enumerate()
            {
                let Some(training_row) = training_row else {
                    continue;
                };
                let row_start = group_position * widest_option_count;
                let option_count = sequence_row.option_marker_positions.len();
                scores_by_row.insert(
                    *training_row,
                    flat_logits[row_start..row_start + option_count].to_vec(),
                );
            }
            reclaim_mlx_buffer_cache(math)?;
        }
        let mut flat_scores = Vec::new();
        for row_number in row_numbers {
            let Some(row_scores) = scores_by_row.get(row_number) else {
                return Err(GpuTrainingError(
                    "a question was left out of every scoring pass".to_owned(),
                ));
            };
            flat_scores.extend_from_slice(row_scores);
        }
        math.upload(&flat_scores, &[flat_scores.len()])
    }

    fn parameter_gradients(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
        row_targets: &RowTargets,
    ) -> GpuResult<Vec<MlxArray>> {
        let row_numbers = &question_batch.dataset_question_row_numbers;
        let mut summed_gradients = Vec::with_capacity(parameters.len());
        for micro_batch_positions in self.micro_batches(row_numbers) {
            let mut micro_row_numbers = Vec::with_capacity(micro_batch_positions.len());
            let mut option_counts = Vec::with_capacity(micro_batch_positions.len());
            for batch_position in &micro_batch_positions {
                let row_number = row_numbers[*batch_position];
                micro_row_numbers.push(row_number);
                option_counts.push(self.sequence_rows[row_number].option_marker_positions.len());
            }
            let micro_batch_loss = LayaMicroBatchLoss {
                loaded_state: Rc::clone(&self.loaded_state),
                sequence_rows: Rc::clone(&self.sequence_rows),
                layout: Rc::clone(&self.layout),
                row_numbers: micro_row_numbers,
                targets: MicroBatchTargets::take(
                    math.runtime,
                    row_targets,
                    &micro_batch_positions,
                    &option_counts,
                )?,
            };
            let differentiated = loss_gradients(micro_batch_loss, parameters)?;
            add_and_evaluate(
                math.runtime,
                &mut summed_gradients,
                differentiated.gradients,
            )?;
            reclaim_mlx_buffer_cache(math)?;
        }
        Ok(summed_gradients)
    }
}

/// One micro-batch's objective, owning everything its graph reads.
struct LayaMicroBatchLoss {
    loaded_state: Rc<LoadedLayaState>,
    sequence_rows: Rc<Vec<SequenceRow>>,
    layout: Rc<WholeModelLayout>,
    row_numbers: Vec<usize>,
    targets: MicroBatchTargets,
}

impl TrainableLoss for LayaMicroBatchLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let trained_parts = self.layout.trained_parts(trainable_arrays)?;
        let mut micro_rows = Vec::with_capacity(self.row_numbers.len());
        for row_number in &self.row_numbers {
            micro_rows.push(&self.sequence_rows[*row_number]);
        }
        let padded_logits = self
            .loaded_state
            .padded_option_logits(
                &forward_rows(&micro_rows),
                &TrainedParts {
                    tuned_tensors: &trained_parts.tuned_tensors,
                    adapters: Some(&trained_parts.adapters),
                },
            )
            .map_err(loss_failure)?;
        self.targets.weighted_cross_entropy(context, &padded_logits)
    }
}

/// Returns MLX's buffer cache to the system once it is worth the GPU wait.
fn reclaim_mlx_buffer_cache(math: GpuMath<'_>) -> GpuResult<()> {
    match math
        .runtime
        .synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
            MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
        ) {
        Ok(()) => Ok(()),
        Err(reclaim_error) => Err(GpuTrainingError(reclaim_error.to_string())),
    }
}

fn scoring_failure(failure: DecisionFailure) -> GpuTrainingError {
    GpuTrainingError(failure.to_string())
}

fn loss_failure(failure: DecisionFailure) -> MlxCError {
    MlxCError {
        operation: "build the whole-model training loss",
        description: failure.to_string(),
    }
}

fn forward_rows<'rows>(sequence_rows: &[&'rows SequenceRow]) -> ForwardRows<'rows> {
    let mut rows = ForwardRows::with_capacity(sequence_rows.len());
    for sequence_row in sequence_rows {
        rows.push(
            &sequence_row.token_ids,
            sequence_row.question_type_index,
            &sequence_row.option_marker_positions,
        );
    }
    rows
}

#[cfg(test)]
#[path = "whole_model_scorer_tests.rs"]
mod tests;
