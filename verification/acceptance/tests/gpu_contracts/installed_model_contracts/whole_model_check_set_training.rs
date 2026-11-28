//! A whole-model training runs to the end on a job of the makeup `astronomical
//! decision train --whole-model` sends: labelled lines plus the command's own
//! 120-question Romeo and Juliet check set, whose answers training must keep.
//! The check set is the command's file, read as the command reads it; the
//! labelled lines are the positive control's, split by position rather than
//! by state as the command splits them, which changes which held-back lines
//! are scored and not the shapes the passes run.
//!
//! The positive control trains on labelled lines alone. The check set adds
//! longer texts and every question kind (yes-or-no, choice and score) to the
//! batches, and with them clears of the MLX buffer cache between passes. On
//! MLX's CUDA backend that job crashed the worker inside `cudaGraphLaunch` a
//! few steps in (issue #2699): with CUDA graphs on, a launch after a clear of
//! the MLX buffer cache crashed, and it ran to the end with graphs off or the
//! clear skipped. The patch `mlx-0.32.5-cuda-graph-cache-before-pool-trim`
//! drops every encoder's cached graph executables at its next commit after a
//! clear. This journey asserts that at least one pass cleared the MLX buffer
//! cache, so it keeps exercising that path.
//!
//! It also holds Laya's recipe (`LAYA_WHOLE_MODEL_TRAINING_SETTINGS`) to both
//! halves of what the command needs: the whole model learns the code on the
//! unseen lines, and keeps most of the check set's answers. Run it with
//! `cargo test-laya-whole-model-check-set-training --configuration PATH`
//! naming Laya's `model_directory`.

use decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, QuestionRole, TrainingExample,
};
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{OrderedJsonValue, parse_system_one_questions};
use decision_training::training_verdict::MINIMUM_CHECK_AGREEMENT;
use model_serving::decision_models::adapter_training::mlx_buffer_cache_reclaim::reclaimed_pass_count;
use serde::Deserialize;
use test_harness::test_run::TestRunContext;

use super::whole_model_positive_control::{
    HeldBackRole, REQUIRED_TRAINED_ACCURACY, coded_job, held_back_accuracy, highest_scoring,
};
use super::whole_model_training::{loaded_engine, train_to_completion};

/// The share of the check set's answers the trained model must keep. Learning
/// a rule the base model lacks moves some answers it was not asked about: at
/// Laya's recipe this job kept 110 of 120 (91.7%), and at a learning rate of 0
/// all 120, so the scoring path itself changes nothing. This floor catches a
/// recipe that drifts further; the verdict's own line, which flags a trained
/// folder to the person, is `MINIMUM_CHECK_AGREEMENT` (95%), and this job sits
/// near it (issue #2705).
const MINIMUM_KEPT_CHECK_ANSWER_SHARE: f64 = 0.9;

/// The check set `astronomical decision train` adds to every job.
const CHECK_SET_TEXT: &str = include_str!(
    "../../../../../apps/astronomical-cli/src/verbs/decision/check_set_romeo_and_juliet.jsonl"
);

/// One check-set line: a state and the questions asked about it.
#[derive(Deserialize)]
struct CheckSetLine {
    state: String,
    questions: OrderedJsonValue,
}

/// The check set as the command sends it: every question with the `Check`
/// role.
fn check_set_examples() -> Vec<TrainingExample> {
    let mut check_examples = Vec::new();
    for check_line_text in CHECK_SET_TEXT.lines() {
        let check_line: CheckSetLine =
            serde_json::from_str(check_line_text).expect("the check set line parses");
        let question_count = parse_system_one_questions(&check_line.questions)
            .expect("the check set questions are well formed")
            .len();
        check_examples.push(TrainingExample {
            state: OrderedJsonValue::Text(check_line.state),
            questions: check_line.questions,
            question_roles: vec![QuestionRole::Check; question_count],
        });
    }
    check_examples
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_train_laya_as_a_whole_on_labelled_lines_with_the_commands_check_set() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let mut decision_engine = loaded_engine(TunableHeadFamily::Laya, &model_directory);
    let mut job = coded_job();
    job.examples.extend(check_set_examples());
    let reclaimed_passes_before = reclaimed_pass_count();
    let outcome = train_to_completion(decision_engine.as_mut(), job.clone(), "check-set-training");
    let reclaimed_passes = reclaimed_pass_count() - reclaimed_passes_before;
    eprintln!(
        "[check-set-training] family=Laya parity_gap={} reclaimed_passes={reclaimed_passes} runs={:?}",
        outcome.largest_parity_gap, outcome.runs
    );
    let (agreeing_answers, check_answers) = check_agreement(&job, &outcome);
    let check_agreement_share = agreeing_answers as f64 / check_answers as f64;
    eprintln!(
        "[check-set-training] check_answers={check_answers} unchanged={agreeing_answers} agreement={check_agreement_share:.3} verdict_would_flag={}",
        check_agreement_share < MINIMUM_CHECK_AGREEMENT
    );
    let (_, choosing_accuracy) = held_back_accuracy(&job, &outcome, HeldBackRole::Choosing);
    let (served_accuracy, trained_accuracy) =
        held_back_accuracy(&job, &outcome, HeldBackRole::Judging);
    eprintln!(
        "[check-set-training] choosing_right_trained={choosing_accuracy:.3} judged_right_served={served_accuracy:.3} judged_right_trained={trained_accuracy:.3}"
    );
    assert_eq!(outcome.largest_parity_gap, 0.0);
    assert!(
        trained_accuracy >= REQUIRED_TRAINED_ACCURACY,
        "the whole model must learn the code with the check set in the job: {trained_accuracy:.3} right on unseen lines"
    );
    assert!(
        check_agreement_share >= MINIMUM_KEPT_CHECK_ANSWER_SHARE,
        "the trained model changed too many of the check set's answers: {agreeing_answers} of {check_answers} unchanged"
    );
    assert!(
        reclaimed_passes > 0,
        "no pass cleared the MLX buffer cache, so the journey no longer exercises the clear"
    );
}

/// How many of the check set's answers the trained model gives as the shipped
/// one did, and how many it scores: a check question's right answer is the
/// shipped model's own.
fn check_agreement(job: &DecisionTrainingJob, outcome: &DecisionTrainingOutcome) -> (usize, usize) {
    let (mut agreeing_answers, mut check_answers) = (0, 0);
    for (example, trained_questions) in job.examples.iter().zip(&outcome.examples) {
        for (question_role, trained_question) in
            example.question_roles.iter().zip(trained_questions)
        {
            if *question_role != QuestionRole::Check || trained_question.trained_logits.is_empty() {
                continue;
            }
            check_answers += 1;
            let shipped_answer = highest_scoring(&trained_question.served.option_logits);
            if highest_scoring(&trained_question.trained_logits) == shipped_answer {
                agreeing_answers += 1;
            }
        }
    }
    (agreeing_answers, check_answers)
}
