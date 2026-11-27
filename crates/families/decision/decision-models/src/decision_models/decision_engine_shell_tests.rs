//! The shell's behaviour around a model that is not loaded, through families
//! that never load one: construction refuses a folder the family rejects,
//! every call that needs the model refuses with the shell's or the family's
//! own wording, and the calls that only tidy up stay quiet. The loaded path
//! is covered by the Laya, Kev and Clef journeys, which need real folders.

use std::path::Path;
use std::rc::Rc;

use decision_contract::decision_training_job::TrainingExample;
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{
    DecisionFailure, DecisionQuestion, DecisionQuestionBody, HeadInputRequest, OrderedJsonValue,
    QuestionOutcome,
};
use decision_training::head_parameters::ParameterLayout;
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use super::{
    AdmittedDecision, DecisionEngineShell, DecisionFamily, LoadedDecisionModel, PreparedFolder,
    admit_questions_then_load, safetensors_bytes_in,
};
use crate::decision_engine::DecisionEngine;
use crate::decision_models::training::question_selection::RowSource;
use crate::decision_models::training::training_session::{DecisionTrainingHost, WholeModelScorer};

/// A family whose folders are usable unless they are named "unusable", and
/// whose model never loads.
struct GlobeFamily;

impl DecisionFamily for GlobeFamily {
    type Prepared = ();
    type LoadedState = ();
    type RequestPlan = ();
    const TUNABLE_FAMILY: TunableHeadFamily = TunableHeadFamily::Kev;

    fn prepare(model_directory: &Path) -> Result<PreparedFolder<()>, DecisionFailure> {
        if model_directory.ends_with("unusable") {
            return Err(DecisionFailure::fatal_execution(
                "the Globe folder is unusable",
            ));
        }
        Ok(PreparedFolder {
            prepared: (),
            resident_weight_bytes: 0,
        })
    }

    fn load_state(
        _model_directory: &Path,
        _prepared: &mut Option<()>,
        _memory_limits: MlxMemoryLimits,
    ) -> Result<LoadedDecisionModel<()>, DecisionFailure> {
        Err(DecisionFailure::fatal_execution("the Globe never loads"))
    }

    fn runtime(_loaded_state: &()) -> &MlxRuntime {
        unreachable!("a Globe model is never loaded")
    }

    fn runtime_mut(_loaded_state: &mut ()) -> &mut MlxRuntime {
        unreachable!("a Globe model is never loaded")
    }

    fn admit_decision<'loaded>(
        loaded_state: Option<&'loaded Rc<()>>,
        _state: &OrderedJsonValue,
        questions: &[DecisionQuestion],
    ) -> Result<AdmittedDecision<'loaded, (), ()>, DecisionFailure> {
        admit_questions_then_load("Globe", loaded_state, questions)
    }

    fn answer_questions(
        _loaded_state: &Rc<()>,
        _request_plan: (),
        _state: &OrderedJsonValue,
        _questions: &[DecisionQuestion],
        _head_input_request: HeadInputRequest,
    ) -> Result<(Vec<QuestionOutcome>, u32), DecisionFailure> {
        Ok((Vec::new(), 0))
    }

    fn whole_model_scorer(
        _loaded_state: &Rc<()>,
        _examples: &[TrainingExample],
        _row_sources: &[RowSource],
        _scorer_layout: &ParameterLayout,
    ) -> Result<WholeModelScorer, DecisionFailure> {
        Err(DecisionFailure::fatal_execution("the Globe cannot train"))
    }
}

fn unloaded_engine() -> DecisionEngineShell<GlobeFamily> {
    DecisionEngineShell::new("Globe-1", Path::new("globe"), 1 << 30, 1 << 20)
        .expect("a usable Globe folder should construct")
}

fn noul_question() -> DecisionQuestion {
    DecisionQuestion {
        name: "is_a_stage".to_owned(),
        instructions: Some(OrderedJsonValue::Text(
            "All the world's a stage.".to_owned(),
        )),
        body: DecisionQuestionBody::Noul {
            false_description: None,
            true_description: None,
            labels: None,
        },
    }
}

fn train_before_load_failure() -> DecisionFailure {
    DecisionFailure::fatal_execution("the decision engine was asked to train before load")
}

#[test]
fn should_refuse_a_folder_the_family_cannot_prepare() {
    let construction =
        DecisionEngineShell::<GlobeFamily>::new("Globe-1", Path::new("unusable"), 1 << 30, 1 << 20);
    assert_eq!(
        construction.err(),
        Some(DecisionFailure::fatal_execution(
            "the Globe folder is unusable"
        ))
    );
}

#[test]
fn should_refuse_to_decide_before_load_in_the_family_wording() {
    let mut engine = unloaded_engine();
    let decision = engine.decide(
        &OrderedJsonValue::Null,
        &[noul_question()],
        HeadInputRequest::Omit,
    );
    assert_eq!(
        decision.err(),
        Some(DecisionFailure::fatal_execution(
            "the Globe engine was asked to decide before load"
        ))
    );
}

#[test]
fn should_refuse_a_request_without_questions_before_checking_the_load() {
    let mut engine = unloaded_engine();
    let decision = engine.decide(&OrderedJsonValue::Null, &[], HeadInputRequest::Omit);
    assert_eq!(
        decision.err(),
        Some(DecisionFailure::invalid_request(
            "a decision request needs at least one question"
        ))
    );
}

#[test]
fn should_refuse_training_work_before_load() {
    let mut engine = unloaded_engine();
    assert_eq!(
        engine.mlx_runtime().err(),
        Some(train_before_load_failure())
    );
    assert_eq!(
        engine
            .whole_model_scorer(&[], &[], &ParameterLayout::new(&[]))
            .err(),
        Some(train_before_load_failure())
    );
    assert_eq!(
        engine.advance_training().err(),
        Some(DecisionFailure::fatal_execution(
            "no training is in progress"
        ))
    );
}

#[test]
fn should_stay_quiet_and_report_no_memory_before_load() {
    let mut engine = unloaded_engine();
    engine.cancel_training();
    assert!(engine.take_post_cleanup_memory_telemetry().is_none());
    assert!(engine.collect_mlx_memory_telemetry().is_none());
    assert_eq!(engine.model_directory(), Path::new("globe"));
    assert_eq!(engine.tunable_family(), TunableHeadFamily::Kev);
}

#[test]
fn should_count_only_the_safetensors_files_directly_in_the_folder() {
    let model_folder = tempfile::tempdir().expect("a scratch folder should be created");
    std::fs::write(model_folder.path().join("model.safetensors"), [0_u8; 10])
        .expect("the weights should be written");
    std::fs::write(model_folder.path().join("head.safetensors"), [0_u8; 7])
        .expect("the head should be written");
    std::fs::write(model_folder.path().join("config.json"), [0_u8; 5])
        .expect("the configuration should be written");
    assert_eq!(safetensors_bytes_in(model_folder.path()), 17);
}

#[test]
fn should_count_no_bytes_in_a_missing_folder() {
    assert_eq!(safetensors_bytes_in(Path::new("no-such-folder")), 0);
}
