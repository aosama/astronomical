//! Scoring and differentiating Kev questions through the whole model, with
//! add-on weights on every dense projection of the backbone and the pointer
//! head trainable.
//!
//! Scores that are compared with serving's (the parity check, the end of
//! each pass, the final scores) run serving's own path: the state prefilled
//! once, each question's row read from it, the pointer head on the CPU,
//! with the add-ons put in force and the head built from the current
//! weights. So what training judges is exactly what the trained folder
//! serves.
//!
//! Gradient steps run many questions through the backbone in one pass
//! (`empty_state_forward`: each question's state and row as one sequence,
//! which computes what serving's state-then-row reading does), with the
//! Gated DeltaNet recurrence's backward pass attached and the pointer head
//! as MLX operations, so MLX traces the objective back to every add-on.
//! Run one question at a time this cost 133 ms a question on Kev 0.8B;
//! batched under the memory-scaled token budget it costs about 52 ms,
//! within about 1.7 times of the model's arithmetic
//! (`cargo test-kev-whole-model-training-speed`).

use std::collections::BTreeMap;
use std::rc::Rc;

use decision_contract::decision_training_job::TrainingExample;
use decision_contract::head_tuning_file::{HeadTuning, TunedTensor};
use decision_contract::{DecisionFailure, parse_system_one_questions};
use decision_training::head_parameters::ParameterLayout;
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};

use super::engine::{LoadedKevState, question_scores};
use super::sequence::{KevQuestionRow, build_kev_sequence};
use crate::decision_models::adapter_training::layer_recomputation::{
    CheckpointedDecoderLayers, PassRecomputation, as_layer_recomputation,
};
use crate::decision_models::adapter_training::loss_gradient::{TrainableLoss, loss_gradients};
use crate::decision_models::adapter_training::micro_batch_objective::{
    MicroBatchTargets, add_and_evaluate,
};
use crate::decision_models::adapter_training::mlx_buffer_cache_reclaim::reclaim_mlx_buffer_cache;
use crate::decision_models::adapter_training::whole_model_layout::{
    ADAPTER_RANK, WholeModelLayout,
};
use crate::decision_models::forward_pass_planning::{
    plan_forward_passes, recomputation_threshold_tokens, training_tokens_per_pass,
};
use crate::decision_models::qwen_backbone::{HoldsDecisionBackbone, backbone_memory_limit_bytes};
use crate::decision_models::training::gpu_math::{GpuMath, GpuResult, GpuTrainingError};
use crate::decision_models::training::question_batch::QuestionBatch;
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::row_objective::RowTargets;
use crate::decision_models::training::trainable_scorer::TrainableScorer;
use crate::decision_models::training::training_session::WholeModelScorer;
use crate::qwen3_5_core::model::layer_additions::Qwen3_5LayerAdditions;
use crate::qwen3_5_core::model_math::differentiable_gated_delta::DifferentiableGatedDelta;

/// What one padded token costs while a micro-batch runs backwards through
/// Kev 0.8B, measured with `cargo test-kev-whole-model-training-speed`:
/// 9.76 GiB at 1,024 tokens and 12.35 GiB at 2,048, about 2.6 MB a token
/// beyond the model, with some room on top.
const KEV_TRAINING_BYTES_PER_PADDED_TOKEN: usize = 3_000_000;

/// One example's tokens: its state, and each question's row.
pub(super) struct KevTrainingExample {
    state_token_ids: Vec<u32>,
    question_rows: Vec<KevQuestionRow>,
}

/// Kev's whole model, as the training loop's scorer.
pub struct KevWholeModelScorer {
    loaded_state: Rc<LoadedKevState>,
    /// Every example of the job, tokenized, by example index.
    examples: Rc<Vec<KevTrainingExample>>,
    /// Each training row's example and question.
    row_sources: Vec<RowSource>,
    layout: Rc<WholeModelLayout>,
    differentiable_recurrence: Rc<DifferentiableGatedDelta>,
    /// The padded tokens one gradient pass may hold.
    tokens_per_pass: usize,
    /// Recomputes the layers of a pass over that budget: one long question.
    pass_recomputation: PassRecomputation,
}

impl KevWholeModelScorer {
    /// The whole model over the questions at `row_sources`, with an add-on
    /// on every dense projection and the pointer head laid out as
    /// `scorer_layout`.
    pub(super) fn for_rows(
        loaded_state: &Rc<LoadedKevState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        let backbone = &loaded_state.backbone;
        if backbone.has_add_ons {
            // Training on top of earlier add-ons would need them as the
            // starting point and as what the anchor pulls toward.
            return Err(DecisionFailure::invalid_request(
                "this model was already trained as a whole; train the original model instead",
            ));
        }
        let targets_by_layer =
            backbone
                .model
                .adapter_targets_by_layer()
                .map_err(|target_error| {
                    DecisionFailure::invalid_request(&format!(
                        "this model cannot train its whole model: {target_error}"
                    ))
                })?;
        let backbone_holder: Rc<dyn HoldsDecisionBackbone> = loaded_state.clone();
        let memory_limit_bytes = backbone_memory_limit_bytes(&loaded_state.backbone)?;
        let pass_recomputation = PassRecomputation::new(
            backbone_holder,
            &targets_by_layer,
            recomputation_threshold_tokens(memory_limit_bytes, KEV_TRAINING_BYTES_PER_PADDED_TOKEN),
        );
        let layout = Rc::new(WholeModelLayout {
            scorer_layout: scorer_layout.clone(),
            adapter_targets: targets_by_layer.into_iter().flatten().collect(),
            adapter_rank: ADAPTER_RANK,
        });
        let mut tokenized_examples = Vec::with_capacity(examples.len());
        for example in examples {
            tokenized_examples.push(tokenized_example(loaded_state, example)?);
        }
        let differentiable_recurrence = DifferentiableGatedDelta::new().map_err(|mlx_error| {
            DecisionFailure::fatal_execution(&format!(
                "the trainable recurrence could not be built: {mlx_error}"
            ))
        })?;
        let scorer = Self {
            pass_recomputation,
            loaded_state: Rc::clone(loaded_state),
            examples: Rc::new(tokenized_examples),
            row_sources: row_sources.to_vec(),
            layout: Rc::clone(&layout),
            differentiable_recurrence: Rc::new(differentiable_recurrence),
            tokens_per_pass: training_tokens_per_pass(
                memory_limit_bytes,
                KEV_TRAINING_BYTES_PER_PADDED_TOKEN,
            ),
        };
        Ok(WholeModelScorer {
            scorer: Box::new(scorer),
            layout,
        })
    }

    /// The batch's positions grouped by example, in first-seen order.
    fn positions_by_example(&self, row_numbers: &[usize]) -> Vec<(usize, Vec<usize>)> {
        let mut example_positions: Vec<(usize, Vec<usize>)> = Vec::new();
        for (batch_position, row_number) in row_numbers.iter().enumerate() {
            let example_index = self.row_sources[*row_number].example_index;
            match example_positions
                .iter_mut()
                .find(|(grouped_example, _)| *grouped_example == example_index)
            {
                Some((_, positions)) => positions.push(batch_position),
                None => example_positions.push((example_index, vec![batch_position])),
            }
        }
        example_positions
    }

    /// The pointer head serving would build from the current scorer weights.
    fn tuned_pointer_head(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
    ) -> GpuResult<super::pointer_head::KevPointerHead> {
        let mut tuned_tensors = Vec::with_capacity(self.layout.scorer_layout.tensors.len());
        for (parameter_tensor, parameter) in
            self.layout.scorer_layout.tensors.iter().zip(parameters)
        {
            tuned_tensors.push(TunedTensor {
                name: parameter_tensor.name.clone(),
                shape: parameter_tensor.shape.clone(),
                values: math.download(parameter)?,
            });
        }
        let mut pointer_head = self.loaded_state.pointer_head.clone();
        pointer_head
            .apply_tuning(&HeadTuning {
                tensors: tuned_tensors,
            })
            .map_err(scoring_failure)?;
        Ok(pointer_head)
    }

    /// Each batch row's served scores with the add-ons in `parameters` in
    /// force, flattened in batch order.
    fn served_scores(
        &self,
        math: GpuMath<'_>,
        row_numbers: &[usize],
        pointer_head: &super::pointer_head::KevPointerHead,
    ) -> GpuResult<Vec<f32>> {
        let backbone = &self.loaded_state.backbone;
        let mut scores_by_position: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
        for (example_index, positions) in self.positions_by_example(row_numbers) {
            let example = &self.examples[example_index];
            let state_length = example.state_token_ids.len();
            let mut decoder_state = backbone.fresh_state().map_err(scoring_failure)?;
            backbone
                .prefill(&example.state_token_ids, 0, &mut decoder_state)
                .map_err(scoring_failure)?;
            for batch_position in positions {
                let question_index = self.row_sources[row_numbers[batch_position]].question_index;
                let question_scores = question_scores(
                    backbone,
                    pointer_head,
                    &example.question_rows[question_index],
                    state_length,
                    &mut decoder_state,
                    decision_contract::HeadInputRequest::Omit,
                )
                .map_err(scoring_failure)?;
                scores_by_position.insert(batch_position, question_scores.option_logits);
            }
            drop(decoder_state);
            reclaim_mlx_buffer_cache(math)?;
        }
        let mut flat_scores = Vec::new();
        for row_scores in scores_by_position.into_values() {
            flat_scores.extend_from_slice(&row_scores);
        }
        Ok(flat_scores)
    }
}

impl TrainableScorer for KevWholeModelScorer {
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        let pointer_head = self.tuned_pointer_head(math, parameters)?;
        let trained_parts = self.layout.trained_parts(parameters)?;
        let model = &self.loaded_state.backbone.model;
        model.set_layer_additions(Some(Rc::new(Qwen3_5LayerAdditions {
            adapters: trained_parts.adapters,
            differentiable_recurrence: None,
        })));
        let served_scores = self.served_scores(
            math,
            &question_batch.dataset_question_row_numbers,
            &pointer_head,
        );
        // Cleared whatever happened: the backbone serves no add-ons of a
        // training's after it.
        model.set_layer_additions(None);
        let flat_scores = served_scores?;
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
        let mut sequence_lengths = Vec::with_capacity(row_numbers.len());
        let mut text_indices = Vec::with_capacity(row_numbers.len());
        for row_number in row_numbers {
            let row_source = self.row_sources[*row_number];
            text_indices.push(row_source.example_index);
            let example = &self.examples[row_source.example_index];
            sequence_lengths.push(
                example.state_token_ids.len()
                    + example.question_rows[row_source.question_index]
                        .token_ids
                        .len(),
            );
        }
        let mut summed_gradients = Vec::with_capacity(parameters.len());
        for pass_positions in plan_forward_passes(&sequence_lengths, self.tokens_per_pass) {
            let layer_recomputation =
                self.pass_recomputation
                    .for_pass(&pass_positions, &sequence_lengths, &text_indices);
            let mut pass_sources = Vec::with_capacity(pass_positions.len());
            let mut option_counts = Vec::with_capacity(pass_positions.len());
            for batch_position in &pass_positions {
                let row_source = self.row_sources[row_numbers[*batch_position]];
                let question_row = &self.examples[row_source.example_index].question_rows
                    [row_source.question_index];
                option_counts.push(question_row.option_closing_offsets.len());
                pass_sources.push(row_source);
            }
            let micro_batch_loss = KevMicroBatchLoss {
                loaded_state: Rc::clone(&self.loaded_state),
                examples: Rc::clone(&self.examples),
                row_sources: pass_sources,
                layout: Rc::clone(&self.layout),
                differentiable_recurrence: Rc::clone(&self.differentiable_recurrence),
                layer_recomputation,
                targets: MicroBatchTargets::take(
                    math.runtime,
                    row_targets,
                    &pass_positions,
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

    fn recomputed_text_count(&self) -> usize {
        self.pass_recomputation.recomputed_text_count()
    }
}

/// One micro-batch's objective, owning everything its graph reads.
struct KevMicroBatchLoss {
    loaded_state: Rc<LoadedKevState>,
    examples: Rc<Vec<KevTrainingExample>>,
    /// Each micro-batch row's example and question.
    row_sources: Vec<RowSource>,
    layout: Rc<WholeModelLayout>,
    differentiable_recurrence: Rc<DifferentiableGatedDelta>,
    targets: MicroBatchTargets,
    /// Set when the pass is over the memory budget.
    layer_recomputation: Option<Rc<CheckpointedDecoderLayers>>,
}

impl TrainableLoss for KevMicroBatchLoss {
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let trained_parts = self.layout.trained_parts(trainable_arrays)?;
        let model = &self.loaded_state.backbone.model;
        model.set_layer_additions(Some(Rc::new(Qwen3_5LayerAdditions {
            adapters: trained_parts.adapters,
            differentiable_recurrence: Some(Rc::clone(&self.differentiable_recurrence)),
        })));
        let padded_logits = self.padded_logits(context, &trained_parts.tuned_tensors);
        model.set_layer_additions(None);
        self.targets
            .weighted_cross_entropy(context, &padded_logits?)
    }
}

/// Where each row's pointer head reads, in the flattened `[rows · longest]`
/// hidden states of one pass.
struct ReadoutPositions {
    decision_positions: Vec<i32>,
    option_positions: Vec<i32>,
    /// Each option's row.
    option_rows: Vec<i32>,
    /// `[rows · widest]`: each slot's option, or one past the last option
    /// for padding.
    padded_option_slots: Vec<i32>,
}

impl KevMicroBatchLoss {
    /// `[rows, widest]` option scores: every row's state and question run
    /// as one sequence in a single pass, and the pointer head applied with
    /// the trainable weights to all rows at once.
    fn padded_logits(
        &self,
        context: &MlxBindingsContext,
        head_tensors: &BTreeMap<String, MlxArray>,
    ) -> Result<MlxArray, MlxCError> {
        let mut token_rows = Vec::with_capacity(self.row_sources.len());
        for row_source in &self.row_sources {
            let example = &self.examples[row_source.example_index];
            let mut token_row = example.state_token_ids.clone();
            token_row
                .extend_from_slice(&example.question_rows[row_source.question_index].token_ids);
            token_rows.push(token_row);
        }
        let mut row_references = Vec::with_capacity(token_rows.len());
        let mut longest_row = 0;
        for token_row in &token_rows {
            row_references.push(token_row.as_slice());
            longest_row = longest_row.max(token_row.len());
        }
        let hidden_states = self
            .loaded_state
            .backbone
            .model
            .final_normalized_hidden_states_of_rows(
                &row_references,
                as_layer_recomputation(self.layer_recomputation.as_ref()),
            )
            .map_err(|forward_error| MlxCError {
                operation: "build Kev's whole-model training loss",
                description: forward_error.to_string(),
            })?;
        let hidden_size = hidden_states.shape()[2];
        let flat_hidden_states = context.reshape(
            &hidden_states,
            &[(token_rows.len() * longest_row) as i32, hidden_size],
        )?;
        let positions = self.readout_positions(longest_row);
        pointer_head_logits(
            context,
            head_tensors,
            &flat_hidden_states,
            &positions,
            self.row_sources.len(),
        )
    }

    fn readout_positions(&self, longest_row: usize) -> ReadoutPositions {
        let widest_option_count = self.targets.widest_option_count();
        let mut positions = ReadoutPositions {
            decision_positions: Vec::with_capacity(self.row_sources.len()),
            option_positions: Vec::new(),
            option_rows: Vec::new(),
            padded_option_slots: Vec::with_capacity(self.row_sources.len() * widest_option_count),
        };
        let mut row_slots = Vec::with_capacity(self.row_sources.len());
        for (row_index, row_source) in self.row_sources.iter().enumerate() {
            let example = &self.examples[row_source.example_index];
            let question_row = &example.question_rows[row_source.question_index];
            let row_start = row_index * longest_row + example.state_token_ids.len();
            positions
                .decision_positions
                .push((row_start + question_row.decision_offset()) as i32);
            let mut slots = Vec::with_capacity(question_row.option_closing_offsets.len());
            for option_closing_offset in &question_row.option_closing_offsets {
                slots.push(positions.option_positions.len() as i32);
                positions
                    .option_positions
                    .push((row_start + option_closing_offset) as i32);
                positions.option_rows.push(row_index as i32);
            }
            row_slots.push(slots);
        }
        let padding_slot = positions.option_positions.len() as i32;
        for slots in row_slots {
            for slot_index in 0..widest_option_count {
                positions
                    .padded_option_slots
                    .push(slots.get(slot_index).copied().unwrap_or(padding_slot));
            }
        }
        positions
    }
}

/// `[rows, widest]` pointer-head scores, as `KevPointerHead::option_logits`
/// computes them on the CPU: `(Wk·option + bk) · (Wq·decision + bq) /
/// √pointer_dimension`, with zero in padding slots.
fn pointer_head_logits(
    context: &MlxBindingsContext,
    head_tensors: &BTreeMap<String, MlxArray>,
    flat_hidden_states: &MlxArray,
    positions: &ReadoutPositions,
    row_count: usize,
) -> Result<MlxArray, MlxCError> {
    let pointer_dimension = head_tensor(head_tensors, "q.bias")?.shape()[0];
    let decision_states = context.take_axis(
        flat_hidden_states,
        &index_array(context, &positions.decision_positions)?,
        0,
    )?;
    let option_states = context.take_axis(
        flat_hidden_states,
        &index_array(context, &positions.option_positions)?,
        0,
    )?;
    let queries = projected(context, head_tensors, "q", &decision_states)?;
    let keys = projected(context, head_tensors, "k", &option_states)?;
    let query_per_option =
        context.take_axis(&queries, &index_array(context, &positions.option_rows)?, 0)?;
    let option_scores = context.multiply_scalar(
        &context.sum_axis(&context.multiply(&keys, &query_per_option)?, 1, false)?,
        1.0 / (pointer_dimension as f32).sqrt(),
    )?;
    padded_by_slot(
        context,
        &option_scores,
        &positions.padded_option_slots,
        row_count,
    )
}

/// `states · Wᵀ + b` with the head's `projection` weights.
fn projected(
    context: &MlxBindingsContext,
    head_tensors: &BTreeMap<String, MlxArray>,
    projection: &str,
    states: &MlxArray,
) -> Result<MlxArray, MlxCError> {
    let weight = head_tensor(head_tensors, &format!("{projection}.weight"))?;
    let bias = head_tensor(head_tensors, &format!("{projection}.bias"))?;
    context.add(
        &context.matmul(states, &context.transpose_axes(weight, &[1, 0])?)?,
        bias,
    )
}

/// `[rows, widest]` from per-option scores and each slot's option, padding
/// slots reading an appended zero.
fn padded_by_slot(
    context: &MlxBindingsContext,
    option_scores: &MlxArray,
    padded_option_slots: &[i32],
    row_count: usize,
) -> Result<MlxArray, MlxCError> {
    let zero_slot = context.zeros(&[1], option_scores.dtype())?;
    let scores_with_zero_slot = context.concatenate_axis(&[option_scores, &zero_slot], 0)?;
    let slots = context.array_from_i32(
        padded_option_slots,
        &[
            row_count as i32,
            (padded_option_slots.len() / row_count.max(1)) as i32,
        ],
    )?;
    context.take_axis(&scores_with_zero_slot, &slots, 0)
}

fn index_array(context: &MlxBindingsContext, indices: &[i32]) -> Result<MlxArray, MlxCError> {
    context.array_from_i32(indices, &[indices.len() as i32])
}

fn head_tensor<'tensors>(
    head_tensors: &'tensors BTreeMap<String, MlxArray>,
    tensor_name: &str,
) -> Result<&'tensors MlxArray, MlxCError> {
    match head_tensors.get(tensor_name) {
        Some(tensor) => Ok(tensor),
        None => Err(MlxCError {
            operation: "build Kev's pointer head",
            description: format!("the scorer has no tensor {tensor_name}"),
        }),
    }
}

/// One example as Kev reads it.
fn tokenized_example(
    loaded_state: &LoadedKevState,
    example: &TrainingExample,
) -> Result<KevTrainingExample, DecisionFailure> {
    let questions = parse_system_one_questions(&example.questions)
        .map_err(|question_error| DecisionFailure::invalid_request(&question_error.to_string()))?;
    let request_sequence = build_kev_sequence(&loaded_state.tokenizer, &example.state, &questions)?;
    Ok(KevTrainingExample {
        state_token_ids: request_sequence.state_token_ids,
        question_rows: request_sequence.question_rows,
    })
}

fn scoring_failure(failure: DecisionFailure) -> GpuTrainingError {
    GpuTrainingError(failure.to_string())
}
