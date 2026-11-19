//! The Laya decision engine over direct MLX arrays.
//!
//! The engine owns a model folder's lifecycle: what it reads at construction
//! (calibration, a trained scorer, add-on weights), loading the artifact,
//! answering requests through the loaded model's forward pass, and any
//! training in progress. The contract crate turns the logits into typed
//! answers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use decision_contract::calibration_file::CalibrationOverride;
use decision_contract::decision_adapters_file::DecisionAdapters;
use decision_contract::decision_training_job::{DecisionTrainingJob, TrainingExample};
use decision_contract::head_tuning_file::{HeadTuning, TunableHeadFamily};
use decision_contract::laya_sequence::LayaQuestionSequence;
use decision_contract::{
    DecisionFailure, DecisionQuestion, DecisionQuestionKind, FloatVector, HeadInputRequest,
    HeadInputs, OrderedJsonValue, QuestionInputCoverage, QuestionOutcome,
    answer_from_probabilities, calibrated_probabilities,
};
use decision_training::head_parameters::ParameterLayout;
use mlx_bridge::MlxArray;
use runtime_integration::{MlxMemoryLimits, MlxRuntime, MlxSafetensors};

use crate::decision_engine::DecisionTrainingStep;
use crate::decision_engine::{DecisionEngine, DecisionEngineLoadResult, DecisionEngineOutput};
use crate::decision_models::calibration_override::read_calibration_override;
use crate::decision_models::decision_adapters::{
    check_adapter_targets, read_decision_adapters, unusable_adapters,
};
use crate::decision_models::head_tuning::{check_tuned_shape, read_head_tuning, unusable_tuning};
use crate::decision_models::laya::adapter_targets::laya_adapter_targets;
use crate::decision_models::laya::artifact::LayaArtifact;
use crate::decision_models::laya::configuration::LayaConfiguration;
use crate::decision_models::laya::head::ScoredHeadRow;
use crate::decision_models::laya::loaded_state::LoadedLayaState;
#[cfg(test)]
use crate::decision_models::laya::loaded_state::sequence_failure;
use crate::decision_models::laya::tokenizer::LayaTokenizer;
use crate::decision_models::laya::whole_model_scorer::LayaWholeModelScorer;
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::{
    DecisionTrainingHost, DecisionTrainingSession, WholeModelScorer, keep_session_if_running,
    read_through_decide, start_session,
};
use crate::linear_adapters::LinearAdapters;
use crate::memory::telemetry::MlxMemoryTelemetry;
use crate::modernbert::weight_layout::{ModernBertWeightLayout, TokenEmbeddingStorage};
use model_primitives::forward_pass_memory::forward_pass_memory_telemetry;

/// The tensor-name prefix a Laya checkpoint keeps its encoder under.
const ENCODER_TENSOR_PREFIX: &str = "encoder";
/// The tensor whose presence means the token embeddings are stored quantized.
const QUANTIZED_EMBEDDING_SCALES_TENSOR: &str = "encoder.embeddings.tok_embeddings.scales";

/// A Laya runtime bound to one artifact directory.
pub struct LayaDecisionEngine {
    model_id: String,
    /// The model folder, which training reads the shipped scorer from.
    model_directory: PathBuf,
    effective_mlx_memory_ceiling_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    resident_weight_bytes: u64,
    preloaded_artifact: Option<LayaArtifact>,
    /// Temperatures fitted on the person's own examples, read at
    /// construction so an unusable file fails before any weights load.
    calibration_override: CalibrationOverride,
    /// The folder's trained scorer, read at construction for the same reason.
    head_tuning: HeadTuning,
    /// The folder's add-on weights, read at construction for the same reason.
    decision_adapters: DecisionAdapters,
    /// Shared, because a whole-model training's loss is built inside an MLX
    /// closure that must own everything it reads.
    loaded_state: Option<Rc<LoadedLayaState>>,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
    /// A training of the final layers in progress, if any.
    training_session: Option<DecisionTrainingSession>,
}

impl LayaDecisionEngine {
    /// Creates an unloaded engine from validated on-disk artifact evidence.
    pub fn new(
        model_id: impl Into<String>,
        model_directory: &Path,
        effective_mlx_memory_ceiling_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
    ) -> Result<Self, DecisionFailure> {
        let artifact = match LayaArtifact::load(model_directory) {
            Ok(artifact) => artifact,
            Err(artifact_error) => {
                return Err(DecisionFailure::fatal_execution(&format!(
                    "decision artifact initialization failed: {artifact_error}"
                )));
            }
        };
        let calibration_override = read_calibration_override(model_directory)?;
        let head_tuning = read_head_tuning(model_directory, TunableHeadFamily::Laya)?;
        let decision_adapters = read_decision_adapters(model_directory)?;
        Ok(Self {
            model_id: model_id.into(),
            model_directory: model_directory.to_path_buf(),
            calibration_override,
            head_tuning,
            decision_adapters,
            effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes,
            resident_weight_bytes: artifact.weights_file_bytes,
            preloaded_artifact: Some(artifact),
            loaded_state: None,
            post_cleanup_memory_telemetry: None,
            training_session: None,
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

impl DecisionEngine for LayaDecisionEngine {
    fn load(&mut self) -> Result<DecisionEngineLoadResult, DecisionFailure> {
        let Some(artifact) = self.preloaded_artifact.take() else {
            return Err(DecisionFailure::fatal_execution(
                "decision engine load was requested twice",
            ));
        };
        let Ok(memory_limits) = MlxMemoryLimits::new(
            self.effective_mlx_memory_ceiling_bytes,
            self.mlx_buffer_cache_limit_bytes,
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
        let tuned_tensors = tuned_head_tensors(&tensors, &self.head_tuning)?;
        let weight_layout = detect_weight_layout(&tensors);
        let adapters = uploaded_adapters(
            &runtime,
            &tensors,
            (&artifact.configuration, &weight_layout),
            &self.decision_adapters,
        )?;
        self.loaded_state = Some(Rc::new(LoadedLayaState {
            weight_layout,
            runtime,
            tensors,
            tuned_tensors,
            adapters,
            configuration: artifact.configuration,
            tokenizer,
        }));
        Ok(DecisionEngineLoadResult {
            model_id: self.model_id.clone(),
            maximum_tokens_per_question,
        })
    }

    fn decide(
        &mut self,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
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
        let forward_result = loaded_state.scored_rows_by_question(
            questions,
            &question_sequences,
            head_input_request,
        );
        // The pass arrays are dropped by now, whether or not a pass failed;
        // synchronizing and clearing the MLX buffer cache returns their memory
        // before the next request.
        let _cleanup_result = loaded_state
            .runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache();
        // Taken after the cleanup, so it holds no `mlx_buffer_cache` bytes and is
        // the state the next request starts from.
        let post_cleanup_memory_telemetry = match loaded_state.runtime.memory_snapshot() {
            Ok(memory_snapshot) => Some(forward_pass_memory_telemetry(
                &memory_snapshot,
                self.effective_mlx_memory_ceiling_bytes as u64,
                self.resident_weight_bytes,
            )),
            Err(_) => None,
        };
        let decision_outcome = match forward_result {
            Ok(scored_rows) => answered_questions(
                &QuestionTemperatures {
                    configuration: &loaded_state.configuration,
                    calibration_override: &self.calibration_override,
                },
                questions,
                &question_sequences,
                scored_rows,
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
        let memory_snapshot = loaded_state.runtime.memory_snapshot().ok()?;
        Some(forward_pass_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes as u64,
            self.resident_weight_bytes,
        ))
    }
}

/// Where a question's temperature comes from: the folder's calibration for
/// a kind it names, otherwise the checkpoint's own table.
struct QuestionTemperatures<'loaded> {
    configuration: &'loaded LayaConfiguration,
    calibration_override: &'loaded CalibrationOverride,
}

impl QuestionTemperatures<'_> {
    fn temperature_for(&self, kind: DecisionQuestionKind, option_count: usize) -> f64 {
        match self
            .calibration_override
            .temperature_for(kind, option_count)
        {
            Some(calibrated_temperature) => calibrated_temperature,
            None => self
                .configuration
                .temperature_table
                .temperature_for(kind, option_count),
        }
    }
}

/// Turns every question's logits into its answer, and counts the tokens read.
fn answered_questions(
    question_temperatures: &QuestionTemperatures<'_>,
    questions: &[DecisionQuestion],
    question_sequences: &[LayaQuestionSequence],
    scored_rows: Vec<ScoredHeadRow>,
) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
    let mut question_outcomes = Vec::with_capacity(questions.len());
    let mut input_token_count = 0_u32;
    for ((question, question_sequence), scored_row) in
        questions.iter().zip(question_sequences).zip(scored_rows)
    {
        input_token_count =
            input_token_count.saturating_add(question_sequence.token_ids.len() as u32);
        question_outcomes.push(question_outcome(
            question_temperatures,
            question,
            question_sequence,
            scored_row,
        )?);
    }
    Ok((question_outcomes, input_token_count))
}

/// Calibrates one question's logits and builds its typed answer.
///
/// Laya lays its options out in canonical order, so the logits need no
/// reordering before the contract crate reads them.
fn question_outcome(
    question_temperatures: &QuestionTemperatures<'_>,
    question: &DecisionQuestion,
    question_sequence: &LayaQuestionSequence,
    scored_row: ScoredHeadRow,
) -> Result<QuestionOutcome, DecisionFailure> {
    let ScoredHeadRow {
        option_logits,
        marker_states,
    } = scored_row;
    for option_logit in &option_logits {
        if !option_logit.is_finite() {
            return Err(DecisionFailure::fatal_execution(&format!(
                "the model produced a non-finite logit for question {:?}",
                question.name
            )));
        }
    }
    let applied_temperature =
        question_temperatures.temperature_for(question.kind(), option_logits.len());
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
        head_inputs: laya_head_inputs(marker_states),
        training_record: None,
    })
}

/// The marker states the scorer read, as head inputs, when they were asked for.
fn laya_head_inputs(marker_states: Option<Vec<Vec<f32>>>) -> Option<HeadInputs> {
    let marker_states = marker_states?;
    let mut marker_vectors = Vec::with_capacity(marker_states.len());
    for marker_state in marker_states {
        marker_vectors.push(FloatVector(marker_state));
    }
    Some(HeadInputs::Laya {
        marker_states: marker_vectors,
    })
}

/// The trained scorer tensors as 32-bit arrays, each checked against the
/// shape of the shipped tensor it replaces.
fn tuned_head_tensors(
    tensors: &MlxSafetensors,
    head_tuning: &HeadTuning,
) -> Result<BTreeMap<String, MlxArray>, DecisionFailure> {
    let mut tuned_tensors = BTreeMap::new();
    for tuned_tensor in &head_tuning.tensors {
        let Ok(shipped_tensor) = tensors.tensor(&tuned_tensor.name) else {
            return Err(unusable_tuning(&format!(
                "the model has no tensor {}",
                tuned_tensor.name
            )));
        };
        let mut shipped_shape = Vec::new();
        let mut array_shape = Vec::new();
        for dimension in shipped_tensor.shape() {
            shipped_shape.push(dimension as usize);
            array_shape.push(dimension);
        }
        check_tuned_shape(tuned_tensor, &shipped_shape)?;
        match MlxArray::from_f32(&tuned_tensor.values, &array_shape) {
            Ok(tuned_array) => {
                tuned_tensors.insert(tuned_tensor.name.clone(), tuned_array);
            }
            Err(array_error) => return Err(unusable_tuning(&array_error.to_string())),
        }
    }
    Ok(tuned_tensors)
}

/// The folder's add-ons on the GPU, each checked against the layer it
/// belongs to; none when the folder has none.
fn uploaded_adapters(
    runtime: &MlxRuntime,
    tensors: &MlxSafetensors,
    (configuration, weight_layout): (&LayaConfiguration, &ModernBertWeightLayout),
    decision_adapters: &DecisionAdapters,
) -> Result<LinearAdapters, DecisionFailure> {
    if decision_adapters.is_empty() {
        return Ok(LinearAdapters::default());
    }
    let targets = laya_adapter_targets(tensors, configuration, weight_layout)?;
    check_adapter_targets(decision_adapters, &targets)?;
    match LinearAdapters::upload(runtime, decision_adapters) {
        Ok(adapters) => Ok(adapters),
        Err(upload_error) => Err(unusable_adapters(&upload_error.to_string())),
    }
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

/// What a training needs from this engine.
impl DecisionTrainingHost for LayaDecisionEngine {
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
            Some(loaded_state) => Ok(&loaded_state.runtime),
            None => Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to train before load",
            )),
        }
    }

    fn model_directory(&self) -> &Path {
        &self.model_directory
    }

    fn tunable_family(&self) -> TunableHeadFamily {
        TunableHeadFamily::Laya
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
        LayaWholeModelScorer::for_rows(loaded_state, examples, row_sources, scorer_layout)
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
