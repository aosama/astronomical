//! The Clef decision engine: one forward over the whole request, then the
//! joint head over every question at once.
//!
//! The state, every question and every option are one sequence, so a
//! request runs the backbone once whatever it asks. The backbone's final
//! states are read for every position, in standard prefill runs, because
//! the head attends to all of them; the head then scores every option of
//! every question together, and each question's logits become probabilities
//! with a plain softmax, as the reference's `systemone` computes them.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use astronomical_decision_contract::clef_answers::clef_answer_from_scored_probabilities;
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, OrderedJsonValue, QuestionInputCoverage, QuestionOutcome,
};
use astronomical_runtime_integration::MlxMemoryLimits;
use mlx_bridge::MlxArray;

use super::head_layers::{ClefHeadLayers, head_math};
use super::head_settings::ClefHeadSettings;
use super::head_weights::ClefHeadWeights;
use super::joint_head::{ClefHeadInputs, SpanLayout};
use super::prompt::{
    CLEF_MAXIMUM_SEQUENCE_TOKENS, ClefQuestionLayout, ClefSequence, build_clef_sequence,
};
use super::tokenizer::ClefTokenizer;
use crate::decision_engine::{DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput};
use crate::decision_models::qwen_backbone::{PREFILL_CHUNK_TOKENS, QwenDecisionBackbone};
use crate::memory::forward_pass_telemetry::forward_pass_memory_telemetry;
use crate::memory::telemetry::MlxMemoryTelemetry;

/// The temperature every Clef logit is used at: the reference applies none.
const CLEF_TEMPERATURE: f64 = 1.0;

/// A Clef runtime bound to one model folder.
pub struct ClefDecisionEngine {
    model_id: String,
    model_directory: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    resident_weight_bytes: u64,
    loaded_state: Option<LoadedClefState>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

struct LoadedClefState {
    backbone: QwenDecisionBackbone,
    tokenizer: ClefTokenizer,
    head_weights: ClefHeadWeights,
    head_settings: ClefHeadSettings,
}

impl ClefDecisionEngine {
    /// Creates an unloaded engine for the Clef folder `model_directory`.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        ClefHeadSettings::read(model_directory)?;
        Ok(Self {
            model_id: model_id.into(),
            model_directory: model_directory.to_path_buf(),
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            resident_weight_bytes: weight_file_bytes(model_directory),
            loaded_state: None,
            post_cleanup_memory_telemetry: None,
        })
    }

    /// The token identifiers of a request's sequence, without running the model.
    ///
    /// Exposed so the layout on the real tokenizer can be compared with the
    /// reference implementation's; serving does not call it.
    #[doc(hidden)]
    pub fn sequence_token_ids(
        &self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<u32>, DecisionFailure> {
        let loaded_state = self.loaded_state()?;
        Ok(build_clef_sequence(&loaded_state.tokenizer, state, questions)?.token_ids)
    }

    fn loaded_state(&self) -> Result<&LoadedClefState, DecisionFailure> {
        self.loaded_state.as_ref().ok_or_else(|| {
            DecisionFailure::fatal_execution("the Clef engine was asked to decide before load")
        })
    }
}

impl DecisionEngine for ClefDecisionEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        if self.loaded_state.is_some() {
            return Err(DecisionFailure::fatal_execution(
                "decision engine load was requested twice",
            ));
        }
        let memory_limits = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.allocator_cache_memory_limit_bytes,
        )
        .map_err(|_| {
            DecisionFailure::fatal_execution("decision runtime memory limits are invalid")
        })?;
        let head_settings = ClefHeadSettings::read(&self.model_directory)?;
        let tokenizer = ClefTokenizer::read(&self.model_directory)?;
        let backbone = QwenDecisionBackbone::load(&self.model_directory, memory_limits)?;
        if backbone.model.hidden_size() as usize != head_settings.hidden_size {
            return Err(DecisionFailure::fatal_execution(
                "Clef's joint head was trained for a different backbone width",
            ));
        }
        let head_weights = ClefHeadWeights::read(
            backbone.model.runtime(),
            &self.model_directory,
            &head_settings,
        )?;
        self.loaded_state = Some(LoadedClefState {
            backbone,
            tokenizer,
            head_weights,
            head_settings,
        });
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question: CLEF_MAXIMUM_SEQUENCE_TOKENS as u32,
        })
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        let started_at = Instant::now();
        if questions.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "a decision request needs at least one question",
            ));
        }
        let loaded_state = self.loaded_state()?;
        let decision_outcome = answer_every_question(loaded_state, state, questions);
        // Whatever happened, the request's decoder state is dropped by now;
        // clearing the allocator cache returns its memory before the next one.
        let runtime = loaded_state.backbone.model.runtime();
        let _cleanup_result = runtime.synchronize_gpu_stream_and_clear_allocator_cache();
        self.post_cleanup_memory_telemetry =
            runtime.memory_snapshot().ok().map(|memory_snapshot| {
                forward_pass_memory_telemetry(
                    &memory_snapshot,
                    self.effective_mlx_memory_ceiling_bytes as u64,
                    self.resident_weight_bytes,
                )
            });
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
        let memory_snapshot = loaded_state
            .backbone
            .model
            .runtime()
            .memory_snapshot()
            .ok()?;
        Some(forward_pass_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes as u64,
            self.resident_weight_bytes,
        ))
    }
}

/// Runs the sequence once and answers every question from the joint head.
fn answer_every_question(
    loaded_state: &LoadedClefState,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
    let sequence = build_clef_sequence(&loaded_state.tokenizer, state, questions)?;
    let hidden_states = whole_sequence_hidden_states(&loaded_state.backbone, &sequence.token_ids)?;
    let option_token_rows = loaded_state
        .backbone
        .model
        .output_embedding_rows(&SpanLayout::option_token_ids(&sequence))
        .map_err(|read_error| {
            DecisionFailure::fatal_execution(&format!(
                "Clef's option embedding read failed: {read_error}"
            ))
        })?;
    let head_layers = ClefHeadLayers {
        runtime: loaded_state.backbone.model.runtime(),
        weights: &loaded_state.head_weights,
        attention_head_count: loaded_state.head_settings.attention_head_count,
    };
    let question_logits = head_layers.joint_option_logits(
        &ClefHeadInputs {
            hidden_states: &hidden_states,
            option_token_rows: &option_token_rows,
            sequence: &sequence,
        },
        loaded_state.head_settings.routing_layer_count,
        loaded_state.head_settings.decoder_layer_count,
    )?;
    let mut question_outcomes = Vec::with_capacity(questions.len());
    for ((question, question_layout), scored_logits) in questions
        .iter()
        .zip(&sequence.questions)
        .zip(&question_logits)
    {
        question_outcomes.push(question_outcome(
            question,
            question_layout,
            scored_logits,
            &sequence,
        )?);
    }
    Ok((
        question_outcomes,
        u32::try_from(sequence.token_ids.len()).unwrap_or(u32::MAX),
    ))
}

/// One question's answer from its logits, in the order Clef scored them.
fn question_outcome(
    question: &DecisionQuestion,
    question_layout: &ClefQuestionLayout,
    scored_logits: &[f64],
    sequence: &ClefSequence,
) -> Result<QuestionOutcome, DecisionFailure> {
    let scored_probabilities = plain_softmax(scored_logits);
    let answer = clef_answer_from_scored_probabilities(
        question,
        &question_layout.question_text,
        &scored_probabilities,
    )
    .map_err(|answer_error| DecisionFailure::fatal_execution(&answer_error.to_string()))?;
    let mut canonical_logits = vec![0.0_f32; scored_logits.len()];
    let mut distinct_option_texts = HashSet::new();
    for (option_text, scored_logit) in question_layout
        .question_text
        .options
        .iter()
        .zip(scored_logits)
    {
        canonical_logits[option_text.canonical_index] = *scored_logit as f32;
        distinct_option_texts.insert(option_text.semantics_text.as_str());
    }
    Ok(QuestionOutcome {
        question_name: question.name.clone(),
        answer,
        option_logits: canonical_logits,
        applied_temperature: CLEF_TEMPERATURE,
        input_coverage: QuestionInputCoverage {
            state_token_count: sequence.state_token_count as u32,
            state_tokens_read: sequence.state_tokens_read as u32,
            option_token_cap: None,
            distinguishable_option_count: distinct_option_texts.len() as u32,
        },
    })
}

/// The final normalized states of every position, `[length, hidden]`, read
/// in standard prefill runs so no one graph spans the whole sequence.
fn whole_sequence_hidden_states(
    backbone: &QwenDecisionBackbone,
    token_ids: &[u32],
) -> Result<MlxArray, DecisionFailure> {
    let mut decoder_state = backbone.fresh_state()?;
    let mut run_hidden_states = Vec::with_capacity(token_ids.len().div_ceil(PREFILL_CHUNK_TOKENS));
    let mut run_start = 0;
    for token_run in token_ids.chunks(PREFILL_CHUNK_TOKENS) {
        run_hidden_states.push(backbone.read_hidden_states(
            token_run,
            run_start as u32,
            &mut decoder_state,
        )?);
        run_start += token_run.len();
    }
    if run_hidden_states.len() == 1 {
        return Ok(run_hidden_states.remove(0));
    }
    let mut run_references = Vec::with_capacity(run_hidden_states.len());
    for run_hidden_state in &run_hidden_states {
        run_references.push(run_hidden_state);
    }
    backbone
        .model
        .runtime()
        .concatenate_axis(&run_references, 0)
        .map_err(head_math)
}

/// `softmax(logits)` in double precision.
fn plain_softmax(option_logits: &[f64]) -> Vec<f64> {
    let largest_logit = option_logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut weights = Vec::with_capacity(option_logits.len());
    let mut weight_total = 0.0_f64;
    for option_logit in option_logits {
        let weight = (option_logit - largest_logit).exp();
        weight_total += weight;
        weights.push(weight);
    }
    for weight in &mut weights {
        *weight /= weight_total;
    }
    weights
}

/// The bytes of the backbone's and head's weight files, reported as resident memory.
fn weight_file_bytes(model_directory: &Path) -> u64 {
    let Ok(directory_entries) = std::fs::read_dir(model_directory) else {
        return 0;
    };
    let mut total_bytes = 0;
    for directory_entry in directory_entries.flatten() {
        let file_name = directory_entry.file_name();
        if file_name.to_string_lossy().ends_with(".safetensors")
            && let Ok(file_metadata) = directory_entry.metadata()
        {
            total_bytes += file_metadata.len();
        }
    }
    total_bytes
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
