//! Whole-model training against a real checkpoint and real serving, for
//! every family that trains its whole model: Laya, Kev and Clef.
//!
//! A whole-model training trains a copy of the model in the worker and
//! writes small add-on weights that serving then reads beside the shipped
//! ones. Unit tests prove each half against oracles; only the real
//! checkpoint proves the two halves agree with each other. This journey
//! checks the three facts the `decision train --whole-model` result rests on:
//!
//! - before any weight moves, the trainable copy scores every question
//!   exactly as serving does (a parity gap of 0);
//! - a folder whose add-ons are all zero serves the shipped scores bit for
//!   bit, so loading add-ons changes nothing by itself;
//! - a folder with the trained add-ons and final layers serves the trained
//!   scores the worker reported, so what training judged is what is served.
//!
//! The examples are a few dozen Romeo and Juliet passages with an arbitrary
//! two-option label: the journey proves agreement, not that training helps
//! (the live positive control in `decision train` proves that).
//!
//! Run one family with its alias and `--configuration PATH`, where the
//! configuration's `model_directory` is that family's model folder:
//! `cargo test-laya-whole-model-training`, `cargo test-kev-whole-model-training`
//! or `cargo test-clef-whole-model-training`.

use std::path::Path;

use astronomical_decision_contract::decision_adapters_file::DecisionAdapters;
use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, QuestionRole, TrainingExample, TrainingScope,
};
use astronomical_decision_contract::head_tuning_file::{HeadTuning, TunableHeadFamily};
use astronomical_decision_contract::trained_tensors::{decision_adapters_of, head_tuning_of};
use astronomical_decision_contract::{
    HeadInputRequest, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_decision_training::calibrated_model_folder::{
    FolderAdditions, FolderCopyProgress, SourceModelFolder, write_calibrated_model_folder,
};
use astronomical_model_serving::{
    ClefDecisionEngine, DecisionEngine, DecisionTrainingStep, KevDecisionEngine, LayaDecisionEngine,
};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

/// Enough for Clef's 9B backbone at 8 bits, its training graph and the
/// `mlx_buffer_cache`; Laya and Kev use a fraction of it.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 20 * 1024 * 1024 * 1024;
const TEST_MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// Enough states for every role to hold several: of each six, four train,
/// one chooses the stopping point and one is judged.
const EXAMPLE_COUNT: usize = 36;
const WORDS_PER_STATE: usize = 40;
const QUESTIONS_JSON: &str = r#"{"mood": {"type": "choice", "instructions": "Which mood does this passage carry?", "criteria": ["calm", "stormy"]}}"#;

/// Swallows copy progress: the folder is small and the journey prints its
/// own lines.
struct SilentCopyProgress;

impl FolderCopyProgress for SilentCopyProgress {
    fn report(&mut self, _processed_bytes: u64, _total_bytes: u64) {}
}

/// One family's engine over a model folder, loaded.
pub(super) fn loaded_engine(
    family: TunableHeadFamily,
    model_directory: &Path,
) -> Box<dyn DecisionEngine> {
    let (ceiling, mlx_buffer_cache_limit) = (
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
    );
    let model_id = "whole-model-training";
    let mut decision_engine: Box<dyn DecisionEngine> = match family {
        TunableHeadFamily::Laya => Box::new(
            LayaDecisionEngine::new(model_id, model_directory, ceiling, mlx_buffer_cache_limit)
                .expect("the Laya artifact should validate"),
        ),
        TunableHeadFamily::Kev => Box::new(
            KevDecisionEngine::new(model_id, model_directory, ceiling, mlx_buffer_cache_limit)
                .expect("the Kev folder should validate"),
        ),
        TunableHeadFamily::Clef => Box::new(
            ClefDecisionEngine::new(model_id, model_directory, ceiling, mlx_buffer_cache_limit)
                .expect("the Clef folder should validate"),
        ),
    };
    decision_engine.load().expect("the checkpoint should load");
    decision_engine
}

fn questions_json() -> OrderedJsonValue {
    serde_json::from_str(QUESTIONS_JSON).expect("the test question is well formed")
}

fn example_role(example_index: usize) -> QuestionRole {
    // The label alternates with the passage, so it is arbitrary to the model.
    let correct_option = example_index % 2;
    match example_index % 6 {
        4 => QuestionRole::Choosing { correct_option },
        5 => QuestionRole::Judging { correct_option },
        _ => QuestionRole::Training { correct_option },
    }
}

fn training_job() -> DecisionTrainingJob {
    let mut examples = Vec::with_capacity(EXAMPLE_COUNT);
    for example_index in 0..EXAMPLE_COUNT {
        examples.push(TrainingExample {
            state: OrderedJsonValue::Text(romeo_and_juliet::passage(
                example_index * WORDS_PER_STATE,
                WORDS_PER_STATE,
            )),
            questions: questions_json(),
            question_roles: vec![example_role(example_index)],
        });
    }
    DecisionTrainingJob {
        examples,
        scope: TrainingScope::WholeModel,
    }
}

fn train_to_completion(
    decision_engine: &mut dyn DecisionEngine,
    job: DecisionTrainingJob,
) -> DecisionTrainingOutcome {
    decision_engine
        .start_training(job)
        .expect("Laya's whole model should be trainable");
    let mut last_progress_line = String::new();
    loop {
        match decision_engine
            .advance_training()
            .expect("every training step should succeed")
        {
            DecisionTrainingStep::Progress {
                stage,
                completed,
                total,
            } => {
                let progress_line = format!("[whole-model] {stage:?} {completed}/{total}");
                if progress_line != last_progress_line {
                    eprintln!("{progress_line}");
                    last_progress_line = progress_line;
                }
            }
            DecisionTrainingStep::Completed(outcome) => return *outcome,
        }
    }
}

/// Each judged example's option scores as `decision_engine` serves them,
/// asking each state alone as `decide` would be asked.
fn served_judging_logits(
    decision_engine: &mut dyn DecisionEngine,
    job: &DecisionTrainingJob,
) -> Vec<Vec<f32>> {
    let questions = parse_system_one_questions(&questions_json()).expect("the question is valid");
    let mut judging_logits = Vec::new();
    for example in &job.examples {
        if !matches!(example.question_roles[0], QuestionRole::Judging { .. }) {
            continue;
        }
        let decision_output = decision_engine
            .decide(&example.state, &questions, HeadInputRequest::Omit)
            .expect("the judged question should be answered");
        judging_logits.push(decision_output.question_outcomes[0].option_logits.clone());
    }
    judging_logits
}

fn trained_judging_logits(
    job: &DecisionTrainingJob,
    training_outcome: &DecisionTrainingOutcome,
) -> Vec<Vec<f32>> {
    let mut judging_logits = Vec::new();
    for (example, trained_questions) in job.examples.iter().zip(&training_outcome.examples) {
        if matches!(example.question_roles[0], QuestionRole::Judging { .. }) {
            judging_logits.push(trained_questions[0].trained_logits.clone());
        }
    }
    judging_logits
}

fn write_folder(
    model_directory: &Path,
    output_directory: &Path,
    head_tuning: Option<(&HeadTuning, TunableHeadFamily)>,
    decision_adapters: &DecisionAdapters,
) {
    write_calibrated_model_folder(
        &SourceModelFolder {
            model_directory,
            other_model_subdirectories: &[],
        },
        output_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning,
            decision_adapters: Some(decision_adapters),
            training_record: None,
        },
        &mut SilentCopyProgress,
    )
    .expect("the trained folder should be written and proven");
}

fn largest_logit_gap(left_logits: &[Vec<f32>], right_logits: &[Vec<f32>]) -> f32 {
    assert_eq!(left_logits.len(), right_logits.len());
    let mut largest_gap = 0.0_f32;
    for (left_question, right_question) in left_logits.iter().zip(right_logits) {
        assert_eq!(left_question.len(), right_question.len());
        for (left_logit, right_logit) in left_question.iter().zip(right_question) {
            largest_gap = largest_gap.max((left_logit - right_logit).abs());
        }
    }
    largest_gap
}

/// Trains `family`'s whole model on the Romeo examples and proves the three
/// facts in this file's header.
async fn prove_whole_model_training(family: TunableHeadFamily) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let job = training_job();

    let mut shipped_engine = loaded_engine(family, &model_directory);
    let shipped_logits = served_judging_logits(shipped_engine.as_mut(), &job);
    let training_outcome = train_to_completion(shipped_engine.as_mut(), job.clone());
    drop(shipped_engine);
    eprintln!(
        "[whole-model] family={family:?} largest_parity_gap={} add_on_tensors={}",
        training_outcome.largest_parity_gap,
        training_outcome.adapter_tensors.len()
    );
    assert_eq!(
        training_outcome.largest_parity_gap, 0.0,
        "the trainable copy must score exactly as serving does"
    );

    let trained_adapters = decision_adapters_of(&training_outcome.adapter_tensors)
        .expect("the add-ons should pair up")
        .expect("a whole-model training returns add-ons");
    let mut zeroed_adapters = trained_adapters.clone();
    for adapter in &mut zeroed_adapters.adapters {
        adapter.up_values.fill(0.0);
    }
    let output_root = tempfile::tempdir().expect("a scratch folder for the written models");

    let zeroed_folder = output_root.path().join("zero-add-ons");
    write_folder(&model_directory, &zeroed_folder, None, &zeroed_adapters);
    let zeroed_logits = served_judging_logits(loaded_engine(family, &zeroed_folder).as_mut(), &job);
    let zeroed_gap = largest_logit_gap(&zeroed_logits, &shipped_logits);
    eprintln!("[whole-model] zero_add_ons_gap_to_shipped={zeroed_gap}");
    assert_eq!(
        zeroed_gap, 0.0,
        "zero add-ons must serve the shipped scores"
    );

    let trained_folder = output_root.path().join("trained");
    let head_tuning = head_tuning_of(&training_outcome.trained_tensors);
    write_folder(
        &model_directory,
        &trained_folder,
        Some((&head_tuning, family)),
        &trained_adapters,
    );
    let served_trained_logits =
        served_judging_logits(loaded_engine(family, &trained_folder).as_mut(), &job);
    let trained_gap = largest_logit_gap(
        &served_trained_logits,
        &trained_judging_logits(&job, &training_outcome),
    );
    eprintln!("[whole-model] trained_folder_gap_to_trained_scores={trained_gap}");
    assert_eq!(
        trained_gap, 0.0,
        "the trained folder must serve the scores training judged"
    );
    assert!(
        largest_logit_gap(&served_trained_logits, &shipped_logits) > 0.0,
        "training must have moved the scores, or the last check proves nothing"
    );
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_serve_shipped_scores_with_zero_add_ons_and_trained_scores_with_trained_ones_on_laya()
 {
    prove_whole_model_training(TunableHeadFamily::Laya).await;
}

#[ignore = "requires a Kev model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_serve_shipped_scores_with_zero_add_ons_and_trained_scores_with_trained_ones_on_kev()
{
    prove_whole_model_training(TunableHeadFamily::Kev).await;
}

#[ignore = "requires a Clef model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_serve_shipped_scores_with_zero_add_ons_and_trained_scores_with_trained_ones_on_clef()
 {
    prove_whole_model_training(TunableHeadFamily::Clef).await;
}
