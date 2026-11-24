//! The Laya decision engine over direct MLX arrays.
//!
//! The engine owns a model folder's lifecycle: what it reads at construction
//! (calibration, a trained scorer, add-on weights), loading the artifact,
//! answering requests through the loaded model's forward pass, and any
//! training in progress. The contract crate turns the logits into typed
//! answers.

use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

use decision_contract::calibration_file::CalibrationOverride;
use decision_contract::decision_adapters_file::DecisionAdapters;
use decision_contract::decision_training_job::TrainingExample;
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

use crate::adapter_targets::laya_adapter_targets;
use crate::artifact::LayaArtifact;
use crate::configuration::LayaConfiguration;
use crate::decision_models::calibration_override::read_calibration_override;
use crate::decision_models::decision_adapters::{
    check_adapter_targets, read_decision_adapters, unusable_adapters,
};
use crate::decision_models::decision_engine_shell::{
    AdmittedDecision, DecisionEngineShell, DecisionFamily, LoadedDecisionModel, PreparedFolder,
};
use crate::decision_models::head_tuning::{check_tuned_shape, read_head_tuning, unusable_tuning};
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::WholeModelScorer;
use crate::head::ScoredHeadRow;
use crate::linear_adapters::LinearAdapters;
use crate::loaded_state::LoadedLayaState;
#[cfg(test)]
use crate::loaded_state::sequence_failure;
use crate::modernbert_trunk::weight_layout::{ModernBertWeightLayout, TokenEmbeddingStorage};
use crate::tokenizer::LayaTokenizer;
use crate::whole_model_scorer::LayaWholeModelScorer;

/// The tensor-name prefix a Laya checkpoint keeps its encoder under.
const ENCODER_TENSOR_PREFIX: &str = "encoder";
/// The tensor whose presence means the token embeddings are stored quantized.
const QUANTIZED_EMBEDDING_SCALES_TENSOR: &str = "encoder.embeddings.tok_embeddings.scales";

/// Laya behind the shared decision engine, bound to one artifact directory.
pub type LayaDecisionEngine = DecisionEngineShell<LayaDecisionFamily>;

pub struct LayaDecisionFamily;

/// What construction reads from a Laya folder, so an unusable file fails
/// before any weights load; load consumes it.
pub struct LayaPreparedFolder {
    artifact: LayaArtifact,
    /// Temperatures fitted on the person's own examples.
    calibration_override: CalibrationOverride,
    /// The folder's trained scorer.
    head_tuning: HeadTuning,
    /// The folder's add-on weights.
    decision_adapters: DecisionAdapters,
}

impl DecisionFamily for LayaDecisionFamily {
    type Prepared = LayaPreparedFolder;
    type LoadedState = LoadedLayaState;
    type RequestPlan = Vec<LayaQuestionSequence>;
    const TUNABLE_FAMILY: TunableHeadFamily = TunableHeadFamily::Laya;

    fn prepare(
        model_directory: &Path,
    ) -> Result<PreparedFolder<LayaPreparedFolder>, DecisionFailure> {
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
        Ok(PreparedFolder {
            resident_weight_bytes: artifact.weights_file_bytes,
            prepared: LayaPreparedFolder {
                artifact,
                calibration_override,
                head_tuning,
                decision_adapters,
            },
        })
    }

    fn load_state(
        _model_directory: &Path,
        prepared: &mut Option<LayaPreparedFolder>,
        memory_limits: MlxMemoryLimits,
    ) -> Result<LoadedDecisionModel<LoadedLayaState>, DecisionFailure> {
        // The shell checks the memory limits before calling here, so a second
        // load after a first one failed on its limits reports the limits
        // again; Laya's own engine took the folder first and then reported
        // the load as requested twice. No caller loads twice.
        let Some(prepared_folder) = prepared.take() else {
            return Err(DecisionFailure::fatal_execution(
                "decision engine load was requested twice",
            ));
        };
        let Ok(runtime) = MlxRuntime::initialize(memory_limits) else {
            return Err(DecisionFailure::fatal_execution(
                "decision runtime initialization failed",
            ));
        };
        load_laya_state(runtime, prepared_folder)
    }

    fn runtime(loaded_state: &LoadedLayaState) -> &MlxRuntime {
        &loaded_state.runtime
    }

    /// Laya checks for a loaded model first, then for questions, and builds
    /// every question's sequence before any model work.
    fn admit_decision<'loaded>(
        loaded_state: Option<&'loaded Rc<LoadedLayaState>>,
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<
        AdmittedDecision<'loaded, LoadedLayaState, Vec<LayaQuestionSequence>>,
        DecisionFailure,
    > {
        let Some(loaded_state) = loaded_state else {
            return Err(DecisionFailure::fatal_execution(
                "the decision engine was asked to decide before load",
            ));
        };
        if questions.is_empty() {
            return Err(DecisionFailure::invalid_request(
                "a decision request needs at least one question",
            ));
        }
        Ok(AdmittedDecision {
            loaded_state,
            request_plan: loaded_state.build_question_sequences(state, questions)?,
        })
    }

    fn answer_questions(
        loaded_state: &Rc<LoadedLayaState>,
        question_sequences: Vec<LayaQuestionSequence>,
        _state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
        let scored_rows = loaded_state.scored_rows_by_question(
            questions,
            &question_sequences,
            head_input_request,
        )?;
        answered_questions(
            &QuestionTemperatures {
                configuration: &loaded_state.configuration,
                calibration_override: &loaded_state.calibration_override,
            },
            questions,
            &question_sequences,
            scored_rows,
        )
    }

    fn whole_model_scorer(
        loaded_state: &Rc<LoadedLayaState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        LayaWholeModelScorer::for_rows(loaded_state, examples, row_sources, scorer_layout)
    }
}

/// Loads the weights, the tokenizer, the trained scorer and the add-ons a
/// prepared folder names, in the order the folder's checks expect.
fn load_laya_state(
    runtime: MlxRuntime,
    prepared_folder: LayaPreparedFolder,
) -> Result<LoadedDecisionModel<LoadedLayaState>, DecisionFailure> {
    let artifact = prepared_folder.artifact;
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
    let tuned_tensors = tuned_head_tensors(&tensors, &prepared_folder.head_tuning)?;
    let weight_layout = detect_weight_layout(&tensors);
    let adapters = uploaded_adapters(
        &runtime,
        &tensors,
        (&artifact.configuration, &weight_layout),
        &prepared_folder.decision_adapters,
    )?;
    Ok(LoadedDecisionModel {
        loaded_state: LoadedLayaState {
            weight_layout,
            runtime,
            tensors,
            tuned_tensors,
            adapters,
            configuration: artifact.configuration,
            tokenizer,
            calibration_override: prepared_folder.calibration_override,
        },
        maximum_tokens_per_question,
    })
}

/// The token sequence each question would be read as, without running the model.
///
/// Exposed so the layout on a real tokenizer can be compared with the
/// reference implementation's; serving does not call it.
/// A free function rather than a method because the engine type is the
/// decision shell, which this family crate does not own.
#[doc(hidden)]
pub fn laya_question_sequences(
    decision_engine: &LayaDecisionEngine,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<Vec<LayaQuestionSequence>, DecisionFailure> {
    let Some(loaded_state) = decision_engine.loaded_state() else {
        return Err(DecisionFailure::fatal_execution(
            "the decision engine was asked for sequences before load",
        ));
    };
    loaded_state.build_question_sequences(state, questions)
}

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

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
