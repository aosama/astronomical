//! The Kev decision engine: one state prefill, then one row per question.
//!
//! The state is prefilled once into the decoder state, which is then
//! checkpointed. Each question's row continues the state from its length,
//! the row's final hidden states are read at the decision token and each
//! option's closing token, and the decoder state is restored before the next
//! question, so no question sees another.

use std::path::Path;
use std::rc::Rc;

use decision_contract::calibration_file::CalibrationOverride;
use decision_contract::decision_training_job::TrainingExample;
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::kev_answers::kev_answer_from_probabilities;
use decision_contract::{
    DecisionFailure, DecisionQuestion, FloatVector, HeadInputRequest, HeadInputs, OrderedJsonValue,
    QuestionInputCoverage, QuestionOutcome, calibrated_probabilities,
};
use decision_training::head_parameters::ParameterLayout;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::pointer_head::{KevPointerHead, kev_probabilities};
use super::sequence::{KEV_MAXIMUM_ROW_TOKENS, KevQuestionRow, KevRowLayout, build_kev_sequence};
use super::settings::KevSettings;
use super::tokenizer::KevTokenizer;
use super::whole_model_scorer::KevWholeModelScorer;
use crate::RequestDecoderStateStack;
use crate::decision_models::calibration_override::read_calibration_override;
use crate::decision_models::decision_engine_shell::{
    AdmittedDecision, DecisionEngineShell, DecisionFamily, LoadedDecisionModel, PreparedFolder,
    admit_questions_then_load, safetensors_bytes_in,
};
use crate::decision_models::head_tuning::read_head_tuning;
use crate::decision_models::qwen_backbone::{HoldsDecisionBackbone, QwenDecisionBackbone};
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::WholeModelScorer;

/// Kev behind the shared decision engine, bound to one converted folder.
pub type KevDecisionEngine = DecisionEngineShell<KevDecisionFamily>;

pub struct KevDecisionFamily;

pub struct LoadedKevState {
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

impl DecisionFamily for KevDecisionFamily {
    type Prepared = ();
    type LoadedState = LoadedKevState;
    type RequestPlan = ();
    const TUNABLE_FAMILY: TunableHeadFamily = TunableHeadFamily::Kev;

    fn prepare(model_directory: &Path) -> Result<PreparedFolder<()>, DecisionFailure> {
        KevSettings::read(model_directory)?;
        // Read here as well as at load so an unusable file is refused before
        // any weights are loaded.
        read_calibration_override(model_directory)?;
        read_head_tuning(model_directory, TunableHeadFamily::Kev)?;
        Ok(PreparedFolder {
            prepared: (),
            resident_weight_bytes: safetensors_bytes_in(model_directory),
        })
    }

    fn load_state(
        model_directory: &Path,
        _prepared: &mut Option<()>,
        memory_limits: MlxMemoryLimits,
    ) -> Result<LoadedDecisionModel<LoadedKevState>, DecisionFailure> {
        let settings = KevSettings::read(model_directory)?;
        let calibration_override = read_calibration_override(model_directory)?;
        let mut pointer_head = KevPointerHead::read(
            model_directory,
            settings.layout.head_file_name(),
            settings.pointer_dimension,
        )?;
        pointer_head.apply_tuning(&read_head_tuning(model_directory, TunableHeadFamily::Kev)?)?;
        let tokenizer = KevTokenizer::read(model_directory)?;
        let backbone = QwenDecisionBackbone::load(model_directory, memory_limits)?;
        if backbone.model.hidden_size() as usize != pointer_head.hidden_size() {
            return Err(DecisionFailure::fatal_execution(
                "Kev's head was trained for a different backbone width",
            ));
        }
        Ok(LoadedDecisionModel {
            loaded_state: LoadedKevState {
                backbone,
                tokenizer,
                pointer_head,
                settings,
                calibration_override,
            },
            maximum_tokens_per_question: KEV_MAXIMUM_ROW_TOKENS as u32,
        })
    }

    fn runtime(loaded_state: &LoadedKevState) -> &MlxRuntime {
        loaded_state.backbone.model.runtime()
    }

    fn runtime_mut(loaded_state: &mut LoadedKevState) -> &mut MlxRuntime {
        &mut loaded_state.backbone.model.base.runtime
    }

    fn admit_decision<'loaded>(
        loaded_state: Option<&'loaded Rc<LoadedKevState>>,
        _state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<AdmittedDecision<'loaded, LoadedKevState, ()>, DecisionFailure> {
        admit_questions_then_load("Kev", loaded_state, questions)
    }

    fn answer_questions(
        loaded_state: &Rc<LoadedKevState>,
        _request_plan: (),
        state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
        head_input_request: HeadInputRequest,
    ) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
        answer_every_question(loaded_state, state, questions, head_input_request)
    }

    fn whole_model_scorer(
        loaded_state: &Rc<LoadedKevState>,
        examples: &[TrainingExample],
        row_sources: &[RowSource],
        scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        KevWholeModelScorer::for_rows(loaded_state, examples, row_sources, scorer_layout)
    }
}

/// Each question's row as the backbone would read it, without running it.
///
/// Exposed so the layout on the real tokenizer can be compared with the
/// reference implementation's; serving does not call it.
/// A free function rather than a method because the engine type is the
/// decision shell, which this family crate does not own.
#[doc(hidden)]
pub fn kev_question_row_layouts(
    decision_engine: &KevDecisionEngine,
    state: &OrderedJsonValue,
    questions: &[DecisionQuestion],
) -> Result<Vec<KevRowLayout>, DecisionFailure> {
    let Some(loaded_state) = decision_engine.loaded_state() else {
        return Err(DecisionFailure::fatal_execution(
            "the Kev engine was asked to decide before load",
        ));
    };
    Ok(build_kev_sequence(&loaded_state.tokenizer, state, questions)?.row_layouts())
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
