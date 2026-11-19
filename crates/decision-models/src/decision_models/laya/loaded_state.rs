//! A loaded Laya model and its forward pass: the tensors, tokenizer and
//! settings one artifact needs, shared by serving and by training.
//!
//! One request becomes one sequence per question. The sequences are grouped
//! into forward passes of bounded size; each pass runs the ModernBERT trunk
//! and the decision head as one lazy graph and reads back one logit per
//! option.

use std::collections::BTreeMap;

use astronomical_decision_contract::laya_rendering::{
    laya_question_type_index, laya_truncates_state_from_the_left,
};
use astronomical_decision_contract::laya_sequence::{
    LayaQuestionSequence, LayaSequenceError, build_laya_question_sequence, encode_laya_state,
};
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, HeadInputRequest, OrderedJsonValue,
};
use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_bridge::MlxArray;

use crate::decision_models::forward_pass_planning::{
    LAYA_SERVING_PADDED_TOKENS_PER_PASS, plan_forward_passes,
};
use crate::decision_models::laya::configuration::LayaConfiguration;
use crate::decision_models::laya::head::{HeadPassRows, LayaDecisionHead, ScoredHeadRow};
use crate::decision_models::laya::tokenizer::LayaTokenizer;
use crate::linear_adapters::LinearAdapters;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::trunk::{ModernBertTrunk, ModernBertTrunkError};
use crate::modernbert::weight_layout::ModernBertWeightLayout;

/// One artifact, loaded.
pub struct LoadedLayaState {
    pub runtime: MlxRuntime,
    pub tensors: MlxSafetensors,
    /// Trained scorer tensors in 32-bit, used in place of the shipped ones.
    pub tuned_tensors: BTreeMap<String, MlxArray>,
    /// Add-on weights from a whole-model training; empty when the folder has none.
    pub adapters: LinearAdapters,
    pub configuration: LayaConfiguration,
    pub weight_layout: ModernBertWeightLayout,
    pub tokenizer: LayaTokenizer,
}

impl LoadedLayaState {
    pub fn build_question_sequences(
        &self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<LayaQuestionSequence>, DecisionFailure> {
        let special_tokens = &self.tokenizer.special_tokens;
        let state_token_ids = encode_laya_state(&self.tokenizer, special_tokens, state)
            .map_err(LayaSequenceError::from)
            .map_err(sequence_failure)?;
        let truncate_state_from_the_left = laya_truncates_state_from_the_left(state);
        let mut question_sequences = Vec::with_capacity(questions.len());
        for question in questions {
            question_sequences.push(
                build_laya_question_sequence(
                    &self.tokenizer,
                    special_tokens,
                    self.configuration.sequence_limits,
                    question,
                    &state_token_ids,
                    truncate_state_from_the_left,
                )
                .map_err(sequence_failure)?,
            );
        }
        Ok(question_sequences)
    }

    /// Runs every forward pass the request needs and returns each question's
    /// option logits, in request order.
    pub fn scored_rows_by_question(
        &self,
        questions: &[DecisionQuestion],
        question_sequences: &[LayaQuestionSequence],
        head_input_request: HeadInputRequest,
    ) -> Result<Vec<ScoredHeadRow>, DecisionFailure> {
        let mut sequence_lengths = Vec::with_capacity(question_sequences.len());
        for question_sequence in question_sequences {
            sequence_lengths.push(question_sequence.token_ids.len());
        }
        let mut scored_rows_by_question = Vec::with_capacity(questions.len());
        for _ in questions {
            scored_rows_by_question.push(None);
        }
        for pass_question_indices in
            plan_forward_passes(&sequence_lengths, LAYA_SERVING_PADDED_TOKENS_PER_PASS)
        {
            let pass_rows = self.run_forward_pass(
                questions,
                question_sequences,
                &pass_question_indices,
                head_input_request,
            )?;
            for (question_index, scored_row) in pass_question_indices.into_iter().zip(pass_rows) {
                scored_rows_by_question[question_index] = Some(scored_row);
            }
        }
        let mut scored_rows = Vec::with_capacity(questions.len());
        for scored_row in scored_rows_by_question {
            let Some(scored_row) = scored_row else {
                return Err(DecisionFailure::fatal_execution(
                    "a question was left out of every forward pass",
                ));
            };
            scored_rows.push(scored_row);
        }
        Ok(scored_rows)
    }

    /// Runs one forward pass over the chosen questions and returns their option logits.
    fn run_forward_pass(
        &self,
        questions: &[DecisionQuestion],
        question_sequences: &[LayaQuestionSequence],
        pass_question_indices: &[usize],
        head_input_request: HeadInputRequest,
    ) -> Result<Vec<ScoredHeadRow>, DecisionFailure> {
        let mut forward_rows = ForwardRows::with_capacity(pass_question_indices.len());
        for question_index in pass_question_indices {
            let question_sequence = &question_sequences[*question_index];
            forward_rows.push(
                &question_sequence.token_ids,
                laya_question_type_index(questions[*question_index].kind()),
                &question_sequence.option_marker_positions,
            );
        }
        let token_batch = self.token_batch(&forward_rows)?;
        let (trunk, head) = self.trunk_and_head(&self.served_parts());
        let trunk_hidden_states = trunk
            .encode_hidden_states(&token_batch)
            .map_err(forward_failure)?;
        head.scored_rows(
            &trunk_hidden_states,
            &forward_rows.head_pass_rows(&token_batch),
            head_input_request,
        )
        .map_err(forward_failure)
    }

    /// `[rows, widest option count]` logits as one lazy graph, with
    /// `trained_parts` in place of the folder's own: what training scores
    /// and differentiates. A row's slots past its own options hold
    /// meaningless values the caller must mask.
    pub fn padded_option_logits(
        &self,
        forward_rows: &ForwardRows<'_>,
        trained_parts: &TrainedParts<'_>,
    ) -> Result<MlxArray, DecisionFailure> {
        let token_batch = self.token_batch(forward_rows)?;
        let (trunk, head) = self.trunk_and_head(trained_parts);
        let trunk_hidden_states = trunk
            .encode_hidden_states(&token_batch)
            .map_err(forward_failure)?;
        head.padded_option_logits(
            &trunk_hidden_states,
            &forward_rows.head_pass_rows(&token_batch),
        )
        .map_err(forward_failure)
    }

    /// The parts the folder serves with: its trained scorer and add-ons.
    pub fn served_parts(&self) -> TrainedParts<'_> {
        TrainedParts {
            tuned_tensors: &self.tuned_tensors,
            // `None` when the folder has none, so an untrained model runs
            // exactly the shipped graph.
            adapters: if self.adapters.is_empty() {
                None
            } else {
                Some(&self.adapters)
            },
        }
    }

    fn token_batch(
        &self,
        forward_rows: &ForwardRows<'_>,
    ) -> Result<PaddedTokenBatch, DecisionFailure> {
        match PaddedTokenBatch::from_sequences(
            &forward_rows.token_sequences,
            self.tokenizer.pad_token_id,
        ) {
            Ok(token_batch) => Ok(token_batch),
            Err(batch_error) => Err(DecisionFailure::fatal_execution(&format!(
                "the question sequences could not be batched: {batch_error}"
            ))),
        }
    }

    fn trunk_and_head<'parts>(
        &'parts self,
        trained_parts: &TrainedParts<'parts>,
    ) -> (ModernBertTrunk<'parts>, LayaDecisionHead<'parts>) {
        let trunk = ModernBertTrunk {
            runtime: &self.runtime,
            tensors: &self.tensors,
            configuration: &self.configuration.encoder,
            weight_layout: &self.weight_layout,
            adapters: trained_parts.adapters,
        };
        let head = LayaDecisionHead {
            runtime: &self.runtime,
            tensors: &self.tensors,
            tuned_tensors: trained_parts.tuned_tensors,
            adapters: trained_parts.adapters,
            configuration: &self.configuration,
        };
        (trunk, head)
    }
}

/// The question rows of one forward pass, as the model reads them.
pub struct ForwardRows<'rows> {
    token_sequences: Vec<&'rows [u32]>,
    question_type_indices: Vec<u32>,
    option_marker_positions: Vec<&'rows [usize]>,
}

impl<'rows> ForwardRows<'rows> {
    pub fn with_capacity(row_count: usize) -> Self {
        Self {
            token_sequences: Vec::with_capacity(row_count),
            question_type_indices: Vec::with_capacity(row_count),
            option_marker_positions: Vec::with_capacity(row_count),
        }
    }

    pub fn push(
        &mut self,
        token_ids: &'rows [u32],
        question_type_index: u32,
        option_marker_positions: &'rows [usize],
    ) {
        self.token_sequences.push(token_ids);
        self.question_type_indices.push(question_type_index);
        self.option_marker_positions.push(option_marker_positions);
    }

    fn head_pass_rows<'pass>(
        &'pass self,
        token_batch: &'pass PaddedTokenBatch,
    ) -> HeadPassRows<'pass> {
        HeadPassRows {
            token_batch,
            question_type_indices: &self.question_type_indices,
            option_marker_positions: &self.option_marker_positions,
        }
    }
}

/// What a forward pass reads in place of the shipped scorer and layers.
pub struct TrainedParts<'parts> {
    /// Scorer tensors by shipped name, in 32-bit.
    pub tuned_tensors: &'parts BTreeMap<String, MlxArray>,
    /// Add-on weights, or `None` for the shipped layers alone.
    pub adapters: Option<&'parts LinearAdapters>,
}

/// A question that cannot be laid out is the request's fault; a tokenizer
/// that cannot encode is the model's.
pub fn sequence_failure(sequence_error: LayaSequenceError) -> DecisionFailure {
    match sequence_error {
        LayaSequenceError::Rendering(_) | LayaSequenceError::OptionsDoNotFit { .. } => {
            DecisionFailure::invalid_request(&sequence_error.to_string())
        }
        LayaSequenceError::TokenEncoding(_) => {
            DecisionFailure::fatal_execution(&sequence_error.to_string())
        }
    }
}

pub fn forward_failure(forward_error: ModernBertTrunkError) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("decision forward pass failed: {forward_error}"))
}
