//! Laya's whole-model training against the real checkpoint and real serving.
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
//! Run it with `cargo test-laya-whole-model-training --configuration PATH`,
//! where the configuration's `model_directory` is the MLX Laya repository
//! folder.

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
use astronomical_model_serving::{DecisionEngine, DecisionTrainingStep, LayaDecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const TEST_MLX_MEMORY_CEILING_BYTES: usize = 6 * 1024 * 1024 * 1024;
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

fn laya_engine(model_directory: &Path) -> LayaDecisionEngine {
    let mut decision_engine = LayaDecisionEngine::new(
        "laya-whole-model-training",
        model_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate");
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
    decision_engine: &mut LayaDecisionEngine,
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
                let progress_line = format!("[laya-whole-model] {stage:?} {completed}/{total}");
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
    decision_engine: &mut LayaDecisionEngine,
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
    head_tuning: Option<&HeadTuning>,
    decision_adapters: &DecisionAdapters,
) {
    let family_head_tuning = match head_tuning {
        Some(head_tuning) => Some((head_tuning, TunableHeadFamily::Laya)),
        None => None,
    };
    write_calibrated_model_folder(
        &SourceModelFolder {
            model_directory,
            other_model_subdirectories: &[],
        },
        output_directory,
        FolderAdditions {
            calibration_override: None,
            head_tuning: family_head_tuning,
            decision_adapters: Some(decision_adapters),
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

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_serve_shipped_scores_with_zero_add_ons_and_trained_scores_with_trained_ones() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let job = training_job();

    let mut shipped_engine = laya_engine(&model_directory);
    let shipped_logits = served_judging_logits(&mut shipped_engine, &job);
    let training_outcome = train_to_completion(&mut shipped_engine, job.clone());
    drop(shipped_engine);
    eprintln!(
        "[laya-whole-model] largest_parity_gap={} add_on_tensors={}",
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

    let zeroed_folder = output_root.path().join("laya-zero-add-ons");
    write_folder(&model_directory, &zeroed_folder, None, &zeroed_adapters);
    let zeroed_logits = served_judging_logits(&mut laya_engine(&zeroed_folder), &job);
    let zeroed_gap = largest_logit_gap(&zeroed_logits, &shipped_logits);
    eprintln!("[laya-whole-model] zero_add_ons_gap_to_shipped={zeroed_gap}");
    assert_eq!(
        zeroed_gap, 0.0,
        "zero add-ons must serve the shipped scores"
    );

    let trained_folder = output_root.path().join("laya-trained");
    let head_tuning = head_tuning_of(&training_outcome.trained_tensors);
    write_folder(
        &model_directory,
        &trained_folder,
        Some(&head_tuning),
        &trained_adapters,
    );
    let served_trained_logits = served_judging_logits(&mut laya_engine(&trained_folder), &job);
    let trained_gap = largest_logit_gap(
        &served_trained_logits,
        &trained_judging_logits(&job, &training_outcome),
    );
    eprintln!("[laya-whole-model] trained_folder_gap_to_trained_scores={trained_gap}");
    assert_eq!(
        trained_gap, 0.0,
        "the trained folder must serve the scores training judged"
    );
    assert!(
        largest_logit_gap(&served_trained_logits, &shipped_logits) > 0.0,
        "training must have moved the scores, or the last check proves nothing"
    );
}
