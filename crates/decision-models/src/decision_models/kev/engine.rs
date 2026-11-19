//! The Kev decision engine: one state prefill, then one row per question.
//!
//! The state is prefilled once into the decoder state, which is then
//! checkpointed. Each question's row continues the state from its length,
//! the row's final hidden states are read at the decision token and each
//! option's closing token, and the decoder state is restored before the next
//! question, so no question sees another.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use astronomical_decision_contract::calibration_file::CalibrationOverride;
use astronomical_decision_contract::decision_training_job::{DecisionTrainingJob, TrainingExample};
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_decision_contract::kev_answers::kev_answer_from_probabilities;
use astronomical_decision_contract::{
    DecisionFailure, DecisionQuestion, FloatVector, HeadInputRequest, HeadInputs, OrderedJsonValue,
    QuestionInputCoverage, QuestionOutcome, calibrated_probabilities,
};
use astronomical_decision_training::head_parameters::ParameterLayout;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::pointer_head::{KevPointerHead, kev_probabilities};
use super::sequence::{KEV_MAXIMUM_ROW_TOKENS, KevQuestionRow, KevRowLayout, build_kev_sequence};
use super::settings::KevSettings;
use super::tokenizer::KevTokenizer;
use super::whole_model_scorer::KevWholeModelScorer;
use crate::RequestDecoderStateStack;
use crate::decision_engine::DecisionTrainingStep;
use crate::decision_engine::{DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput};
use crate::decision_models::calibration_override::read_calibration_override;
use crate::decision_models::head_tuning::read_head_tuning;
use crate::decision_models::qwen_backbone::{HoldsDecisionBackbone, QwenDecisionBackbone};
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::{
    DecisionTrainingHost, DecisionTrainingSession, WholeModelScorer, keep_session_if_running,
    read_through_decide, start_session,
};
use crate::memory::telemetry::MlxMemoryTelemetry;
use astronomical_model_primitives::forward_pass_memory::forward_pass_memory_telemetry;

/// A Kev runtime bound to one converted folder.
pub struct KevDecisionEngine {
    model_id: String,
    model_directory: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    resident_weight_bytes: u64,
    /// Shared with a whole-model training's scorer, which reads the
    /// backbone and the head from inside MLX's gradient closures.
    loaded_state: Option<Rc<LoadedKevState>>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
    /// A training of the final layers in progress, if any.
    training_session: Option<DecisionTrainingSession>,
}

pub(super) struct LoadedKevState {
    pub(super) backbone: QwenDecisionBackbone,
    pub(super) tokenizer: KevTokenizer,
    pub(super) pointer_head: KevPointerHead,
    settings: KevSettings,
    /// Temperatures fitted on the person's own examples, for the kinds it names.
    calibration_override: CalibrationOverride,
}

impl HoldsDecisionBackbone for LoadedKevState {
    fn decision_backbone(&self) -> &QwenDecisionBackbone {
        &self.backbone
    }
}

impl KevDecisionEngine {
    /// Creates an unloaded engine for the converted folder `model_directory`.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        KevSettings::read(model_directory)?;
        // Read here as well as at load so an unusable file is refused before
        // any weights are loaded.
        read_calibration_override(model_directory)?;
        read_head_tuning(model_directory, TunableHeadFamily::Kev)?;
        Ok(Self {
            model_id: model_id.into(),
            model_directory: model_directory.to_path_buf(),
            effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes,
            resident_weight_bytes: weight_file_bytes(model_directory),
            loaded_state: None,
            post_cleanup_memory_telemetry: None,
            training_session: None,
        })
    }

    /// Each question's row as the backbone would read it, without running it.
    ///
    /// Exposed so the layout on the real tokenizer can be compared with the
    /// reference implementation's; serving does not call it.
    #[doc(hidden)]
    pub fn question_row_layouts(
        &self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<Vec<KevRowLayout>, DecisionFailure> {
        let loaded_state = self.loaded_state()?;
        Ok(build_kev_sequence(&loaded_state.tokenizer, state, questions)?.row_layouts())
    }

    fn loaded_state(&self) -> Result<&LoadedKevState, DecisionFailure> {
        match self.loaded_state.as_deref() {
            Some(loaded_state) => Ok(loaded_state),
            None => Err(DecisionFailure::fatal_execution(
                "the Kev engine was asked to decide before load",
            )),
        }
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
            self.mlx_buffer_cache_limit_bytes,
        )
        .map_err(|_| {
            DecisionFailure::fatal_execution("decision runtime memory limits are invalid")
        })?;
        let settings = KevSettings::read(&self.model_directory)?;
        let calibration_override = read_calibration_override(&self.model_directory)?;
        let mut pointer_head =
            KevPointerHead::read(&self.model_directory, settings.pointer_dimension)?;
        pointer_head.apply_tuning(&read_head_tuning(
            &self.model_directory,
            TunableHeadFamily::Kev,
        )?)?;
        let tokenizer = KevTokenizer::read(&self.model_directory)?;
        let backbone = QwenDecisionBackbone::load(&self.model_directory, memory_limits)?;
        if backbone.model.hidden_size() as usize != pointer_head.hidden_size() {
            return Err(DecisionFailure::fatal_execution(
                "Kev's head was trained for a different backbone width",
            ));
        }
        self.loaded_state = Some(Rc::new(LoadedKevState {
            backbone,
            tokenizer,
            pointer_head,
            settings,
            calibration_override,
        }));
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question: KEV_MAXIMUM_ROW_TOKENS as u32,
        })
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<DecisionEngineOutput, DecisionFailure> {
        let started_at = Instant::now();
        if questions.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "a decision request needs at least one question",
            ));
        }
        let loaded_state = self.loaded_state()?;
        let decision_outcome =
            answer_every_question(loaded_state, state, questions, head_input_request);
        // Whatever happened, the request's decoder state is dropped by now;
        // clearing the MLX buffer cache returns its memory before the next one.
        let runtime = loaded_state.backbone.model.runtime();
        let _cleanup_result = runtime.synchronize_gpu_stream_and_clear_mlx_buffer_cache();
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

    fn start_training(&mut self, job: DecisionTrainingJob) -> Result<(), DecisionFailure> {
        if self.loaded_state.is_none() {
            return Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to train before load",
            ));
        }
        start_session(&mut self.training_session, job)
    }

    fn advance_training(&mut self) -> Result<DecisionTrainingStep, DecisionFailure> {
        let Some(mut session) = self.training_session.take() else {
            return Err(DecisionFailure::fatal_execution(
                "no training is in progress",
            ));
        };
        let step = session.advance(self);
        keep_session_if_running(&mut self.training_session, session, &step);
        step
    }

    fn cancel_training(&mut self) {
        self.training_session = None;
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

/// One question's probabilities and the temperature they were taken at.
///
/// A kind the folder's calibration names uses the shared, clamped softmax
/// every family applies to fitted temperatures. Any other kind keeps Kev's
/// own behaviour: its shipped temperature, used as the checkpoint gives it.
fn kev_calibrated_probabilities(
    loaded_state: &LoadedKevState,
    question: &DecisionQuestion,
    option_logits: &[f32],
) -> (Vec<f64>, f64) {
    let calibrated_temperature = loaded_state
        .calibration_override
        .temperature_for(question.kind(), option_logits.len());
    match calibrated_temperature {
        Some(temperature) => (
            calibrated_probabilities(option_logits, temperature),
            temperature,
        ),
        None => {
            let shipped_temperature = loaded_state.settings.temperature;
            (
                kev_probabilities(option_logits, shipped_temperature),
                shipped_temperature,
            )
        }
    }
}

/// Prefills the state once and answers each question from it.
fn answer_every_question(
    loaded_state: &LoadedKevState,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
    head_input_request: HeadInputRequest,
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
        let QuestionScores {
            option_logits,
            head_inputs,
        } = question_scores(
            &loaded_state.backbone,
            &loaded_state.pointer_head,
            question_row,
            state_length,
            &mut decoder_state,
            head_input_request,
        )?;
        let (option_probabilities, applied_temperature) =
            kev_calibrated_probabilities(loaded_state, question, &option_logits);
        let answer = kev_answer_from_probabilities(question, &option_probabilities)
            .map_err(|answer_error| DecisionFailure::fatal_execution(&answer_error.to_string()))?;
        input_token_count += question_row.token_ids.len();
        question_outcomes.push(QuestionOutcome {
            question_name: question.name.clone(),
            answer,
            option_logits,
            applied_temperature,
            input_coverage: QuestionInputCoverage {
                state_token_count: state_length as u32,
                state_tokens_read: state_length as u32,
                option_token_cap: None,
                distinguishable_option_count: question_row.distinguishable_option_count as u32,
            },
            head_inputs,
            training_record: None,
        });
    }
    Ok((
        question_outcomes,
        u32::try_from(input_token_count).unwrap_or(u32::MAX),
    ))
}

/// One question's option logits and, when asked, what the pointer head read.
pub(super) struct QuestionScores {
    pub(super) option_logits: Vec<f32>,
    head_inputs: Option<HeadInputs>,
}

/// Runs one question's row from the shared state and scores its options
/// with `pointer_head`; the state is restored afterwards, whatever the
/// outcome. A whole-model training scores with this very function, its
/// add-ons in force and its head's current weights, so what it judges is
/// what the trained folder serves.
pub(super) fn question_scores(
    backbone: &QwenDecisionBackbone,
    pointer_head: &KevPointerHead,
    question_row: &KevQuestionRow,
    state_length: usize,
    decoder_state: &mut RequestDecoderStateStack,
    head_input_request: HeadInputRequest,
) -> Result<QuestionScores, DecisionFailure> {
    let state_checkpoint = decoder_state.checkpoint().map_err(|state_error| {
        DecisionFailure::fatal_execution(&format!("the state checkpoint failed: {state_error}"))
    })?;
    let row_reading =
        backbone.read_hidden_states(&question_row.token_ids, state_length as u32, decoder_state);
    decoder_state
        .restore_checkpoint(state_checkpoint)
        .map_err(|state_error| {
            DecisionFailure::fatal_execution(&format!("the state restore failed: {state_error}"))
        })?;
    let row_hidden_states = row_reading?.to_vec_f32().map_err(|read_error| {
        DecisionFailure::fatal_execution(&format!("the readout failed: {read_error}"))
    })?;
    let hidden_size = pointer_head.hidden_size();
    let mut option_hidden_states = Vec::with_capacity(question_row.option_closing_offsets.len());
    for option_closing_offset in &question_row.option_closing_offsets {
        option_hidden_states.push(hidden_state_at(
            &row_hidden_states,
            *option_closing_offset,
            hidden_size,
        ));
    }
    let decision_hidden_state = hidden_state_at(
        &row_hidden_states,
        question_row.decision_offset(),
        hidden_size,
    );
    let option_logits = pointer_head.option_logits(decision_hidden_state, &option_hidden_states);
    let head_inputs = match head_input_request {
        HeadInputRequest::Omit => None,
        HeadInputRequest::Include => Some(kev_head_inputs(
            decision_hidden_state,
            &option_hidden_states,
        )),
    };
    Ok(QuestionScores {
        option_logits,
        head_inputs,
    })
}

/// The exact vectors the pointer head read, copied out for training.
fn kev_head_inputs(decision_hidden_state: &[f32], option_hidden_states: &[&[f32]]) -> HeadInputs {
    let mut option_states = Vec::with_capacity(option_hidden_states.len());
    for option_hidden_state in option_hidden_states {
        option_states.push(FloatVector(option_hidden_state.to_vec()));
    }
    HeadInputs::Kev {
        decision_state: FloatVector(decision_hidden_state.to_vec()),
        option_states,
    }
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

/// What a training needs from this engine.
impl DecisionTrainingHost for KevDecisionEngine {
    fn read_answers(
        &mut self,
        state: &OrderedJsonValue,
        questions: &OrderedJsonValue,
        head_input_request: HeadInputRequest,
    ) -> Result<Vec<QuestionOutcome>, DecisionFailure> {
        read_through_decide(self, state, questions, head_input_request)
    }

    fn mlx_runtime(&self) -> Result<&MlxRuntime, DecisionFailure> {
        match self.loaded_state.as_ref() {
            Some(loaded_state) => Ok(loaded_state.backbone.model.runtime()),
            None => Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to train before load",
            )),
        }
    }

    fn model_directory(&self) -> &Path {
        &self.model_directory
    }

    fn tunable_family(&self) -> TunableHeadFamily {
        TunableHeadFamily::Kev
    }

    fn whole_model_scorer(
        &mut self,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        let Some(loaded_state) = self.loaded_state.as_ref() else {
            return Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to train before load",
            ));
        };
        KevWholeModelScorer::for_rows(loaded_state, examples, row_sources, scorer_layout)
    }
}
