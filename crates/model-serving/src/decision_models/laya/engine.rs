//! The Laya decision engine over direct MLX arrays.
//!
//! One request becomes one sequence per question. The sequences are grouped
//! into forward passes of bounded size; each pass runs the ModernBERT trunk
//! and the decision head as one lazy graph and reads back one logit per
//! option. The contract crate then calibrates the logits into typed answers.

use std::path::Path;
use std::time::Instant;

use astronomical_decision_contract::laya_rendering::{
    laya_question_type_index, laya_truncates_state_from_the_left,
};
use astronomical_decision_contract::laya_sequence::{
    LayaQuestionSequence, LayaSequenceError, build_laya_question_sequence, encode_laya_state,
};
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, OrderedJsonValue, QuestionInputCoverage, QuestionOutcome,
    answer_from_probabilities, calibrated_probabilities,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};

use crate::decision_engine::{DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput};
use crate::decision_models::laya::artifact::LayaArtifact;
use crate::decision_models::laya::batching::{PADDED_TOKENS_PER_PASS, plan_forward_passes};
use crate::decision_models::laya::configuration::LayaConfiguration;
use crate::decision_models::laya::head::{HeadPassRows, LayaDecisionHead};
use crate::decision_models::laya::memory_telemetry::laya_memory_telemetry;
use crate::decision_models::laya::tokenizer::LayaTokenizer;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::modernbert::token_batch::PaddedTokenBatch;
use crate::modernbert::trunk::{ModernBertTrunk, ModernBertTrunkError};
use crate::modernbert::weight_layout::{ModernBertWeightLayout, TokenEmbeddingStorage};

/// The tensor-name prefix a Laya checkpoint keeps its encoder under.
const ENCODER_TENSOR_PREFIX: &str = "encoder";
/// The tensor whose presence means the token embeddings are stored quantized.
const QUANTIZED_EMBEDDING_SCALES_TENSOR: &str = "encoder.embeddings.tok_embeddings.scales";

/// A Laya runtime bound to one artifact directory.
pub struct LayaDecisionEngine {
    model_id: String,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    resident_weight_bytes: u64,
    preloaded_artifact: Option<LayaArtifact>,
    loaded_state: Option<LoadedLayaState>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

struct LoadedLayaState {
    runtime: MlxRuntime,
    tensors: MlxSafetensors,
    configuration: LayaConfiguration,
    weight_layout: ModernBertWeightLayout,
    tokenizer: LayaTokenizer,
}

impl LayaDecisionEngine {
    /// Creates an unloaded engine from validated on-disk artifact evidence.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        let artifact = match LayaArtifact::load(model_directory) {
            Ok(artifact) => artifact,
            Err(artifact_error) => {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "decision artifact initialization failed: {artifact_error}"
                )));
            }
        };
        Ok(Self {
            model_id: model_id.into(),
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            resident_weight_bytes: artifact.weights_file_bytes,
            preloaded_artifact: Some(artifact),
            loaded_state: None,
            post_cleanup_memory_telemetry: None,
        })
    }

    /// The token sequence each question would be read as, without running the model.
    ///
    /// Exposed so the layout on a real tokenizer can be compared with the
    /// reference implementation's; serving does not call it.
    #[doc(hidden)]
    pub fn question_sequences(
        &self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<LayaQuestionSequence>, DecisionFailure> {
        let Some(loaded_state) = self.loaded_state.as_ref() else {
            return Err(DecisionFailure::fatal_execution(
                "the decision engine was asked for sequences before load",
            ));
        };
        loaded_state.build_question_sequences(state, questions)
    }
}

impl LoadedLayaState {
    fn build_question_sequences(
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
    fn option_logits_by_question(
        &self,
        questions: &[DecisionQuestion],
        question_sequences: &[LayaQuestionSequence],
    ) -> Result<Vec<Vec<f32>>, DecisionFailure> {
        let mut sequence_lengths = Vec::with_capacity(question_sequences.len());
        for question_sequence in question_sequences {
            sequence_lengths.push(question_sequence.token_ids.len());
        }
        let mut option_logits_by_question: Vec<Vec<f32>> = vec![Vec::new(); questions.len()];
        for pass_question_indices in plan_forward_passes(&sequence_lengths, PADDED_TOKENS_PER_PASS)
        {
            let pass_logits =
                self.run_forward_pass(questions, question_sequences, &pass_question_indices)?;
            for (question_index, question_logits) in
                pass_question_indices.into_iter().zip(pass_logits)
            {
                option_logits_by_question[question_index] = question_logits;
            }
        }
        Ok(option_logits_by_question)
    }

    /// Runs one forward pass over the chosen questions and returns their option logits.
    fn run_forward_pass(
        &self,
        questions: &[DecisionQuestion],
        question_sequences: &[LayaQuestionSequence],
        pass_question_indices: &[usize],
    ) -> Result<Vec<Vec<f32>>, DecisionFailure> {
        let mut token_sequences = Vec::with_capacity(pass_question_indices.len());
        let mut question_type_indices = Vec::with_capacity(pass_question_indices.len());
        let mut option_marker_positions = Vec::with_capacity(pass_question_indices.len());
        for question_index in pass_question_indices {
            let question_sequence = &question_sequences[*question_index];
            token_sequences.push(question_sequence.token_ids.as_slice());
            question_type_indices.push(laya_question_type_index(questions[*question_index].kind()));
            option_marker_positions.push(question_sequence.option_marker_positions.as_slice());
        }
        let token_batch =
            match PaddedTokenBatch::from_sequences(&token_sequences, self.tokenizer.pad_token_id) {
                Ok(token_batch) => token_batch,
                Err(batch_error) => {
                    return Err(DecisionFailure::fatal_execution(&format!(
                        "the question sequences could not be batched: {batch_error}"
                    )));
                }
            };
        let trunk = ModernBertTrunk {
            runtime: &self.runtime,
            tensors: &self.tensors,
            configuration: &self.configuration.encoder,
            weight_layout: &self.weight_layout,
        };
        let head = LayaDecisionHead {
            runtime: &self.runtime,
            tensors: &self.tensors,
            configuration: &self.configuration,
        };
        let trunk_hidden_states = trunk
            .encode_hidden_states(&token_batch)
            .map_err(forward_failure)?;
        head.option_logits(
            &trunk_hidden_states,
            &HeadPassRows {
                token_batch: &token_batch,
                question_type_indices: &question_type_indices,
                option_marker_positions: &option_marker_positions,
            },
        )
        .map_err(forward_failure)
    }
}

impl DecisionEngine for LayaDecisionEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        let Some(artifact) = self.preloaded_artifact.take() else {
            return Err(DecisionFailure::fatal_execution(
                "decision engine load was requested twice",
            ));
        };
        let Ok(memory_limits) = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes,
        ) else {
            return Err(DecisionFailure::fatal_execution(
                "decision runtime memory limits are invalid",
            ));
        };
        let Ok(runtime) = MlxRuntime::initialize(memory_limits) else {
            return Err(DecisionFailure::fatal_execution(
                "decision runtime initialization failed",
            ));
        };
        let tensors = match runtime.load_safetensors(artifact.weights_file, None) {
            Ok(tensors) => tensors,
            Err(load_error) => {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "decision weights failed to load: {load_error}"
                )));
            }
        };
        let tokenizer = match LayaTokenizer::from_artifact_bytes(
            &artifact.tokenizer_json_bytes,
            &artifact.tokenizer_settings_bytes,
        ) {
            Ok(tokenizer) => tokenizer,
            Err(tokenizer_error) => {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "decision tokenizer failed to load: {tokenizer_error}"
                )));
            }
        };
        let maximum_tokens_per_question = artifact
            .configuration
            .sequence_limits
            .maximum_sequence_tokens as u32;
        self.loaded_state = Some(LoadedLayaState {
            weight_layout: detect_weight_layout(&tensors),
            runtime,
            tensors,
            configuration: artifact.configuration,
            tokenizer,
        });
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question,
        })
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        let started_at = Instant::now();
        let Some(loaded_state) = self.loaded_state.as_ref() else {
            return Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to decide before load",
            ));
        };
        if questions.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "a decision request needs at least one question",
            ));
        }
        let question_sequences = loaded_state.build_question_sequences(state, questions)?;
        let forward_result = loaded_state.option_logits_by_question(questions, &question_sequences);
        // The pass arrays are dropped by now, whether or not a pass failed;
        // synchronizing and clearing the allocator cache returns their memory
        // before the next request.
        let _cleanup_result = loaded_state
            .runtime
            .synchronize_gpu_stream_and_clear_allocator_cache();
        // Taken after the cleanup, so it holds no reusable-cache bytes and is
        // the state the next request starts from.
        let post_cleanup_memory_telemetry = match loaded_state.runtime.memory_snapshot() {
            Ok(memory_snapshot) => Some(laya_memory_telemetry(
                &memory_snapshot,
                self.effective_mlx_memory_ceiling_bytes as u64,
                self.resident_weight_bytes,
            )),
            Err(_) => None,
        };
        let decision_outcome = match forward_result {
            Ok(option_logits_by_question) => answered_questions(
                &loaded_state.configuration,
                questions,
                &question_sequences,
                option_logits_by_question,
            ),
            Err(forward_failure) => Err(forward_failure),
        };
        self.post_cleanup_memory_telemetry = post_cleanup_memory_telemetry;
        let (question_outcomes, input_token_count) = decision_outcome?;
        Ok(DecisionEngineOutput {
            question_outcomes,
            input_token_count,
            elapsed_millis: u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX),
        })
    }

    fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }

    fn collect_mlx_memory_telemetry(&self) -> Option<MlxMemoryTelemetry> {
        let loaded_state = self.loaded_state.as_ref()?;
        let memory_snapshot = loaded_state.runtime.memory_snapshot().ok()?;
        Some(laya_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes as u64,
            self.resident_weight_bytes,
        ))
    }
}

/// Turns every question's logits into its answer, and counts the tokens read.
fn answered_questions(
    configuration: &LayaConfiguration,
    questions: &[DecisionQuestion],
    question_sequences: &[LayaQuestionSequence],
    option_logits_by_question: Vec<Vec<f32>>,
) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
    let mut question_outcomes = Vec::with_capacity(questions.len());
    let mut input_token_count = 0_u32;
    for ((question, question_sequence), option_logits) in questions
        .iter()
        .zip(question_sequences)
        .zip(option_logits_by_question)
    {
        input_token_count =
            input_token_count.saturating_add(question_sequence.token_ids.len() as u32);
        question_outcomes.push(question_outcome(
            configuration,
            question,
            question_sequence,
            option_logits,
        )?);
    }
    Ok((question_outcomes, input_token_count))
}

/// Calibrates one question's logits and builds its typed answer.
///
/// Laya lays its options out in canonical order, so the logits need no
/// reordering before the contract crate reads them.
fn question_outcome(
    configuration: &LayaConfiguration,
    question: &DecisionQuestion,
    question_sequence: &LayaQuestionSequence,
    option_logits: Vec<f32>,
) -> Result<QuestionOutcome, DecisionFailure> {
    for option_logit in &option_logits {
        if !option_logit.is_finite() {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the model produced a non-finite logit for question {:?}",
                question.name
            )));
        }
    }
    let applied_temperature = configuration
        .temperature_table
        .temperature_for(question.kind(), option_logits.len());
    let option_probabilities = calibrated_probabilities(&option_logits, applied_temperature);
    let answer = match answer_from_probabilities(question, &option_probabilities) {
        Ok(answer) => answer,
        Err(answer_error) => {
            return Err(DecisionFailure::fatal_execution(&answer_error.to_string()));
        }
    };
    Ok(QuestionOutcome {
        question_name: question.name.clone(),
        answer,
        option_logits,
        applied_temperature,
        input_coverage: QuestionInputCoverage {
            state_token_count: question_sequence.state_token_count as u32,
            state_tokens_read: question_sequence.state_tokens_used as u32,
            option_token_cap: question_sequence
                .option_token_cap
                .map(|token_cap| token_cap as u32),
            distinguishable_option_count: question_sequence.distinct_option_count as u32,
        },
    })
}

/// Reads how the artifact stores its token embeddings from which tensors it has.
///
/// A quantized table comes with a `scales` tensor and a dense one does not.
/// Asking for the tensor only looks its name up; no weight bytes are read.
fn detect_weight_layout(tensors: &MlxSafetensors) -> ModernBertWeightLayout {
    let token_embedding_storage = if tensors.tensor(QUANTIZED_EMBEDDING_SCALES_TENSOR).is_ok() {
        TokenEmbeddingStorage::AffineQuantized
    } else {
        TokenEmbeddingStorage::Dense
    };
    ModernBertWeightLayout::new(ENCODER_TENSOR_PREFIX, token_embedding_storage)
}

/// A question that cannot be laid out is the request's fault; a tokenizer
/// that cannot encode is the model's.
fn sequence_failure(sequence_error: LayaSequenceError) -> DecisionFailure {
    match sequence_error {
        LayaSequenceError::Rendering(_) | LayaSequenceError::OptionsDoNotFit { .. } => {
            DecisionFailure::invalid_request(&sequence_error.to_string())
        }
        LayaSequenceError::TokenEncoding(_) => {
            DecisionFailure::fatal_execution(&sequence_error.to_string())
        }
    }
}

fn forward_failure(forward_error: ModernBertTrunkError) -> DecisionFailure {
    DecisionFailure::fatal_execution(&format!("decision forward pass failed: {forward_error}"))
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
