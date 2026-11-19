//! Scoring and differentiating Clef questions through the whole model, with
//! add-on weights on every dense projection of the backbone and the joint
//! head's final layers trainable.
//!
//! Scores that are compared with serving's run serving's own path
//! (`answer_every_question`): the add-ons put in force and the head built
//! from the current weights, so what training judges is exactly what the
//! trained folder serves.
//!
//! Gradient steps run several examples' whole sequences through the
//! backbone in one pass (`empty_state_forward`), with the Gated DeltaNet
//! recurrence's backward pass attached, then each example's joint head up
//! to its per-option terms (`option_term_arrays`, the head's own code) and
//! the trainable final mix (`gpu_heads::clef_option_scores`, the formula
//! final-layer training uses). On Clef's 9B a question costs about 440 ms
//! forwards and backwards, within about 1.2 times of its arithmetic
//! (`cargo test-clef-whole-model-training-speed`).

use std::rc::Rc;

use astronomical_decision_contract::decision_training_job::TrainingExample;
use astronomical_decision_contract::head_tuning_file::{HeadTuning, TunedTensor};
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, HeadInputRequest, parse_system_one_questions,
};
use astronomical_decision_training::head_parameters::ParameterLayout;
use mlx_bridge::{MlxArray, MlxBindingsContext, MlxCError};

use super::engine::{LoadedClefState, answer_every_question};
use super::head_layers::{ClefAttentionRoute, ClefHeadLayers};
use super::head_weights::ClefHeadWeights;
use super::joint_head::{ClefHeadInputs, SpanLayout};
use super::prompt::{ClefSequence, build_clef_sequence};
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
use crate::decision_models::training::gpu_heads::clef_option_scores;
use crate::decision_models::training::gpu_math::{GpuMath, GpuResult, GpuTrainingError};
use crate::decision_models::training::question_batch::QuestionBatch;
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::row_objective::RowTargets;
use crate::decision_models::training::trainable_scorer::TrainableScorer;
use crate::decision_models::training::training_session::WholeModelScorer;
use crate::qwen3_5_core::model::layer_additions::Qwen3_5LayerAdditions;
use crate::qwen3_5_core::model_math::differentiable_gated_delta::DifferentiableGatedDelta;

/// What one padded token costs while a micro-batch runs backwards through
/// Clef's 9B, measured with `cargo test-clef-whole-model-training-speed`:
/// 12.78 GiB at 256 tokens and 16.70 GiB at 512, about 16 MB a token,
/// with some room on top. On a 48 GB Mac a pass holds about 500 tokens.
const CLEF_TRAINING_BYTES_PER_PADDED_TOKEN: usize = 17_000_000;

/// One example as Clef reads it.
struct ClefTrainingExample {
    example: TrainingExample,
    questions: Vec<DecisionQuestion>,
    sequence: ClefSequence,
}

/// Clef's whole model, as the training loop's scorer.
pub struct ClefWholeModelScorer {
    loaded_state: Rc<LoadedClefState>,
    examples: Rc<Vec<ClefTrainingExample>>,
    row_sources: Vec<RowSource>,
    layout: Rc<WholeModelLayout>,
    differentiable_recurrence: Rc<DifferentiableGatedDelta>,
    /// The padded tokens one gradient pass may hold.
    tokens_per_pass: usize,
    /// Recomputes the layers of a pass over that budget: one long text.
    pass_recomputation: PassRecomputation,
}

impl ClefWholeModelScorer {
    /// The whole model over the questions at `row_sources`, with an add-on
    /// on every dense projection and the final layers laid out as
    /// `scorer_layout`.
    pub(super) fn for_rows(
        loaded_state: &Rc<LoadedClefState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        let backbone = &loaded_state.backbone;
        if backbone.has_add_ons {
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
            recomputation_threshold_tokens(
                memory_limit_bytes,
                CLEF_TRAINING_BYTES_PER_PADDED_TOKEN,
            ),
        );
        let layout = Rc::new(WholeModelLayout {
            scorer_layout: scorer_layout.clone(),
            adapter_targets: targets_by_layer.into_iter().flatten().collect(),
            adapter_rank: ADAPTER_RANK,
        });
        let mut read_examples = Vec::with_capacity(examples.len());
        for example in examples {
            let questions =
                parse_system_one_questions(&example.questions).map_err(|question_error| {
                    DecisionFailure::invalid_request(&question_error.to_string())
                })?;
            let sequence =
                build_clef_sequence(&loaded_state.tokenizer, &example.state, &questions)?;
            read_examples.push(ClefTrainingExample {
                example: example.clone(),
                questions,
                sequence,
            });
        }
        let differentiable_recurrence = DifferentiableGatedDelta::new().map_err(|mlx_error| {
            DecisionFailure::fatal_execution(&format!(
                "the trainable recurrence could not be built: {mlx_error}"
            ))
        })?;
        let scorer = Self {
            pass_recomputation,
            loaded_state: Rc::clone(loaded_state),
            examples: Rc::new(read_examples),
            row_sources: row_sources.to_vec(),
            layout: Rc::clone(&layout),
            differentiable_recurrence: Rc::new(differentiable_recurrence),
            tokens_per_pass: training_tokens_per_pass(
                memory_limit_bytes,
                CLEF_TRAINING_BYTES_PER_PADDED_TOKEN,
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

    /// The head serving would build from the current final-layer weights.
    fn tuned_head_weights(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
    ) -> GpuResult<ClefHeadWeights> {
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
        self.loaded_state
            .head_weights
            .with_tuning(&HeadTuning {
                tensors: tuned_tensors,
            })
            .map_err(scoring_failure)
    }

    /// Each batch row's served scores, in canonical option order, flattened
    /// in batch order; the add-ons must already be in force.
    fn served_scores(
        &self,
        math: GpuMath<'_>,
        row_numbers: &[usize],
        head_weights: &ClefHeadWeights,
    ) -> GpuResult<Vec<f32>> {
        let mut scores_by_position = vec![Vec::new(); row_numbers.len()];
        for (example_index, positions) in self.positions_by_example(row_numbers) {
            let read_example = &self.examples[example_index];
            let (question_outcomes, _) = answer_every_question(
                &self.loaded_state,
                head_weights,
                &read_example.example.state,
                &read_example.questions,
                HeadInputRequest::Omit,
            )
            .map_err(scoring_failure)?;
            for batch_position in positions {
                let question_index = self.row_sources[row_numbers[batch_position]].question_index;
                scores_by_position[batch_position]
                    .clone_from(&question_outcomes[question_index].option_logits);
            }
            reclaim_mlx_buffer_cache(math)?;
        }
        Ok(scores_by_position.concat())
    }
}

impl TrainableScorer for ClefWholeModelScorer {
    fn option_scores(
        &self,
        math: GpuMath<'_>,
        parameters: &[MlxArray],
        question_batch: &QuestionBatch,
    ) -> GpuResult<MlxArray> {
        let head_weights = self.tuned_head_weights(math, parameters)?;
        let trained_parts = self.layout.trained_parts(parameters)?;
        let model = &self.loaded_state.backbone.model;
        model.set_layer_additions(Some(Rc::new(Qwen3_5LayerAdditions {
            adapters: trained_parts.adapters,
            differentiable_recurrence: None,
        })));
        let served_scores = self.served_scores(
            math,
            &question_batch.dataset_question_row_numbers,
            &head_weights,
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
        let example_groups = self.positions_by_example(row_numbers);
        let mut sequence_lengths = Vec::with_capacity(example_groups.len());
        let mut text_indices = Vec::with_capacity(example_groups.len());
        for (example_index, _) in &example_groups {
            sequence_lengths.push(self.examples[*example_index].sequence.token_ids.len());
            text_indices.push(*example_index);
        }
        let mut summed_gradients = Vec::with_capacity(parameters.len());
        for pass_groups in plan_forward_passes(&sequence_lengths, self.tokens_per_pass) {
            let layer_recomputation =
                self.pass_recomputation
                    .for_pass(&pass_groups, &sequence_lengths, &text_indices);
            let mut pass_examples = Vec::with_capacity(pass_groups.len());
            let mut pass_positions = Vec::new();
            let mut option_counts = Vec::new();
            for group_index in pass_groups {
                let (example_index, positions) = &example_groups[group_index];
                let sequence = &self.examples[*example_index].sequence;
                let mut question_indices = Vec::with_capacity(positions.len());
                for batch_position in positions {
                    let question_index =
                        self.row_sources[row_numbers[*batch_position]].question_index;
                    question_indices.push(question_index);
                    option_counts.push(sequence.questions[question_index].option_spans.len());
                    pass_positions.push(*batch_position);
                }
                pass_examples.push((*example_index, question_indices));
            }
            let micro_batch_loss = ClefMicroBatchLoss {
                loaded_state: Rc::clone(&self.loaded_state),
                examples: Rc::clone(&self.examples),
                pass_examples,
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
struct ClefMicroBatchLoss {
    loaded_state: Rc<LoadedClefState>,
    examples: Rc<Vec<ClefTrainingExample>>,
    /// Each example in the pass, with the questions of it the batch trains.
    pass_examples: Vec<(usize, Vec<usize>)>,
    layout: Rc<WholeModelLayout>,
    differentiable_recurrence: Rc<DifferentiableGatedDelta>,
    /// Set when the pass is over the memory budget.
    layer_recomputation: Option<Rc<CheckpointedDecoderLayers>>,
    targets: MicroBatchTargets,
}

impl TrainableLoss for ClefMicroBatchLoss {
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
        let scorer_count = self.layout.scorer_layout.tensors.len();
        let padded_logits = self.padded_logits(context, &trainable_arrays[..scorer_count]);
        model.set_layer_additions(None);
        self.targets
            .weighted_cross_entropy(context, &padded_logits?)
    }
}

impl ClefMicroBatchLoss {
    /// `[questions, widest]` scores of every trained question in the pass:
    /// all the pass's sequences through the backbone at once, then each
    /// example's joint head on its own rows.
    fn padded_logits(
        &self,
        context: &MlxBindingsContext,
        scorer_parameters: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let mut token_rows = Vec::with_capacity(self.pass_examples.len());
        for (example_index, _) in &self.pass_examples {
            token_rows.push(self.examples[*example_index].sequence.token_ids.as_slice());
        }
        let model = &self.loaded_state.backbone.model;
        let pass_hidden_states = model
            .final_normalized_hidden_states_of_rows(
                &token_rows,
                as_layer_recomputation(self.layer_recomputation.as_ref()),
            )
            .map_err(|forward_error| {
                loss_failure(DecisionFailure::fatal_execution(&forward_error.to_string()))
            })?;
        let hidden_size = pass_hidden_states.shape()[2];
        let mut example_logits = Vec::with_capacity(self.pass_examples.len());
        for (row_index, (example_index, question_indices)) in self.pass_examples.iter().enumerate()
        {
            let sequence = &self.examples[*example_index].sequence;
            let row_states = context.slice(
                &pass_hidden_states,
                &[row_index as i32, 0, 0],
                &[
                    row_index as i32 + 1,
                    sequence.token_ids.len() as i32,
                    hidden_size,
                ],
                &[1, 1, 1],
            )?;
            let hidden_states =
                context.reshape(&row_states, &[sequence.token_ids.len() as i32, hidden_size])?;
            let scores = self.scoring_order_scores(sequence, &hidden_states, scorer_parameters)?;
            example_logits.push(self.canonical_padded_logits(
                context,
                sequence,
                question_indices,
                &scores,
            )?);
        }
        let mut logit_references = Vec::with_capacity(example_logits.len());
        for logits in &example_logits {
            logit_references.push(logits);
        }
        context.concatenate_axis(&logit_references, 0)
    }

    /// Every option's score of one example in the order Clef scores them,
    /// `[options]`: the joint head to its per-option terms, and the
    /// trainable final mix.
    fn scoring_order_scores(
        &self,
        sequence: &ClefSequence,
        hidden_states: &MlxArray,
        scorer_parameters: &[MlxArray],
    ) -> Result<MlxArray, MlxCError> {
        let loaded_state = &self.loaded_state;
        let runtime = loaded_state.backbone.model.runtime();
        // The options' output-embedding rows are shipped weights no add-on
        // touches: the same constant serving reads.
        let option_token_rows = loaded_state
            .backbone
            .model
            .output_embedding_rows(&SpanLayout::option_token_ids(sequence))
            .map_err(|read_error| {
                loss_failure(DecisionFailure::fatal_execution(&read_error.to_string()))
            })?;
        let head_layers = ClefHeadLayers {
            runtime,
            weights: &loaded_state.head_weights,
            attention_head_count: loaded_state.head_settings.attention_head_count,
            attention_route: ClefAttentionRoute::PlainOperations,
        };
        let term_arrays = head_layers
            .option_term_arrays(
                &ClefHeadInputs {
                    hidden_states,
                    option_token_rows: &option_token_rows,
                    sequence,
                },
                loaded_state.head_settings.routing_layer_count,
                loaded_state.head_settings.decoder_layer_count,
            )
            .map_err(loss_failure)?;
        let residual_features = term_arrays
            .residual_features(runtime)
            .map_err(loss_failure)?;
        clef_option_scores(
            GpuMath { runtime },
            scorer_parameters,
            (
                &residual_features,
                &term_arrays.lexical_prior,
                &term_arrays.cosine,
            ),
        )
        .map_err(gpu_failure)
    }

    /// `[questions, widest]` scores of `question_indices`, each row in
    /// canonical option order (the order the right answers are given in),
    /// padding slots reading an appended zero.
    fn canonical_padded_logits(
        &self,
        context: &MlxBindingsContext,
        sequence: &ClefSequence,
        question_indices: &[usize],
        scoring_order_scores: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let widest_option_count = self.targets.widest_option_count();
        let total_option_count = scoring_order_scores.shape()[0];
        let mut first_option_of_question = Vec::with_capacity(sequence.questions.len());
        let mut next_option = 0;
        for question_layout in &sequence.questions {
            first_option_of_question.push(next_option);
            next_option += question_layout.option_spans.len();
        }
        let mut gather_positions =
            vec![total_option_count; question_indices.len() * widest_option_count];
        for (row_index, question_index) in question_indices.iter().enumerate() {
            let question_text = &sequence.questions[*question_index].question_text;
            for (scoring_index, option_text) in question_text.options.iter().enumerate() {
                gather_positions[row_index * widest_option_count + option_text.canonical_index] =
                    (first_option_of_question[*question_index] + scoring_index) as i32;
            }
        }
        let zero_slot = context.zeros(&[1], scoring_order_scores.dtype())?;
        let scores_with_zero_slot =
            context.concatenate_axis(&[scoring_order_scores, &zero_slot], 0)?;
        let positions = context.array_from_i32(
            &gather_positions,
            &[question_indices.len() as i32, widest_option_count as i32],
        )?;
        context.take_axis(&scores_with_zero_slot, &positions, 0)
    }
}

fn scoring_failure(failure: DecisionFailure) -> GpuTrainingError {
    GpuTrainingError(failure.to_string())
}

fn loss_failure(failure: DecisionFailure) -> MlxCError {
    MlxCError {
        operation: "build Clef's whole-model training loss",
        description: failure.to_string(),
    }
}

fn gpu_failure(failure: GpuTrainingError) -> MlxCError {
    MlxCError {
        operation: "build Clef's whole-model training loss",
        description: failure.0,
    }
}
