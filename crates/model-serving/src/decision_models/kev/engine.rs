//! The Kev decision engine: one state prefill, then one row per question.
//!
//! The state is prefilled once into the decoder state, which is then
//! checkpointed. Each question's row continues the state from its length,
//! the row's final hidden states are read at the decision token and each
//! option's closing token, and the decoder state is restored before the next
//! question, so no question sees another.

use std::path::{Path, PathBuf};
use std::time::Instant;

use astronomical_decision_contract::kev_rendering::kev_answer_from_probabilities;
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, OrderedJsonValue, QuestionInputCoverage, QuestionOutcome,
};
use astronomical_runtime_integration::MlxMemoryLimits;

use super::pointer_head::{KevPointerHead, kev_probabilities};
use super::sequence::{KEV_MAXIMUM_ROW_TOKENS, KevQuestionRow, build_kev_sequence};
use super::settings::KevSettings;
use super::tokenizer::KevTokenizer;
use crate::RequestDecoderStateStack;
use crate::decision_engine::{DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput};
use crate::decision_models::memory_telemetry::decision_memory_telemetry;
use crate::decision_models::qwen_backbone::QwenDecisionBackbone;
use crate::memory::telemetry::MlxMemoryTelemetry;

/// A Kev runtime bound to one converted folder.
pub struct KevDecisionEngine {
    model_id: String,
    model_directory: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    resident_weight_bytes: u64,
    loaded_state: Option<LoadedKevState>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

struct LoadedKevState {
    backbone: QwenDecisionBackbone,
    tokenizer: KevTokenizer,
    pointer_head: KevPointerHead,
    settings: KevSettings,
}

impl KevDecisionEngine {
    /// Creates an unloaded engine for the converted folder `model_directory`.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        KevSettings::read(model_directory)?;
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

    fn loaded_state(&self) -> Result<&LoadedKevState, DecisionFailure> {
        self.loaded_state.as_ref().ok_or_else(|| {
            DecisionFailure::fatal_execution("the Kev engine was asked to decide before load")
        })
    }
}

impl DecisionEngine for KevDecisionEngine {
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
        let settings = KevSettings::read(&self.model_directory)?;
        let pointer_head = KevPointerHead::read(&self.model_directory, settings.pointer_dimension)?;
        let tokenizer = KevTokenizer::read(&self.model_directory)?;
        let backbone = QwenDecisionBackbone::load(&self.model_directory, memory_limits)?;
        if backbone.model.hidden_size() as usize != pointer_head.hidden_size() {
            return Err(DecisionFailure::fatal_execution(
                "Kev's head was trained for a different backbone width",
            ));
        }
        self.loaded_state = Some(LoadedKevState {
            backbone,
            tokenizer,
            pointer_head,
            settings,
        });
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question: KEV_MAXIMUM_ROW_TOKENS as u32,
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
                decision_memory_telemetry(
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
        Some(decision_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes as u64,
            self.resident_weight_bytes,
        ))
    }
}

/// Prefills the state once and answers each question from it.
fn answer_every_question(
    loaded_state: &LoadedKevState,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
    let request_sequence = build_kev_sequence(&loaded_state.tokenizer, state, questions)?;
    let state_length = request_sequence.state_token_ids.len();
    let mut decoder_state = loaded_state.backbone.fresh_state()?;
    loaded_state
        .backbone
        .prefill(&request_sequence.state_token_ids, 0, &mut decoder_state)?;
    let mut question_outcomes = Vec::with_capacity(questions.len());
    let mut input_token_count = state_length;
    for (question, question_row) in questions.iter().zip(&request_sequence.question_rows) {
        let option_logits =
            question_option_logits(loaded_state, question_row, state_length, &mut decoder_state)?;
        let option_probabilities =
            kev_probabilities(&option_logits, loaded_state.settings.temperature);
        let answer = kev_answer_from_probabilities(question, &option_probabilities)
            .map_err(|answer_error| DecisionFailure::fatal_execution(&answer_error.to_string()))?;
        input_token_count += question_row.token_ids.len();
        question_outcomes.push(QuestionOutcome {
            question_name: question.name.clone(),
            answer,
            option_logits,
            applied_temperature: loaded_state.settings.temperature,
            input_coverage: QuestionInputCoverage {
                state_token_count: state_length as u32,
                state_tokens_read: state_length as u32,
                option_token_cap: None,
                distinguishable_option_count: question_row.distinguishable_option_count as u32,
            },
        });
    }
    Ok((
        question_outcomes,
        u32::try_from(input_token_count).unwrap_or(u32::MAX),
    ))
}

/// Runs one question's row from the shared state and scores its options;
/// the state is restored afterwards, whatever the outcome.
fn question_option_logits(
    loaded_state: &LoadedKevState,
    question_row: &KevQuestionRow,
    state_length: usize,
    decoder_state: &mut RequestDecoderStateStack,
) -> Result<Vec<f32>, DecisionFailure> {
    let state_checkpoint = decoder_state.checkpoint().map_err(|state_error| {
        DecisionFailure::fatal_execution(&format!("the state checkpoint failed: {state_error}"))
    })?;
    let row_reading = loaded_state.backbone.read_hidden_states(
        &question_row.token_ids,
        state_length as u32,
        decoder_state,
    );
    decoder_state
        .restore_checkpoint(state_checkpoint)
        .map_err(|state_error| {
            DecisionFailure::fatal_execution(&format!("the state restore failed: {state_error}"))
        })?;
    let row_hidden_states = row_reading?.to_vec_f32().map_err(|read_error| {
        DecisionFailure::fatal_execution(&format!("the readout failed: {read_error}"))
    })?;
    let hidden_size = loaded_state.pointer_head.hidden_size();
    let mut option_hidden_states = Vec::with_capacity(question_row.option_closing_offsets.len());
    for option_closing_offset in &question_row.option_closing_offsets {
        option_hidden_states.push(hidden_state_at(
            &row_hidden_states,
            *option_closing_offset,
            hidden_size,
        ));
    }
    Ok(loaded_state.pointer_head.option_logits(
        hidden_state_at(
            &row_hidden_states,
            question_row.decision_offset(),
            hidden_size,
        ),
        &option_hidden_states,
    ))
}

/// The hidden state of the row's token at `row_offset`.
fn hidden_state_at(row_hidden_states: &[f32], row_offset: usize, hidden_size: usize) -> &[f32] {
    &row_hidden_states[row_offset * hidden_size..(row_offset + 1) * hidden_size]
}

/// The bytes of the backbone's weight files, reported as resident memory.
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
