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
//! cache, so it keeps exercising that path. Run it with `cargo
//! test-laya-whole-model-check-set-training --configuration PATH` naming
//! Laya's `model_directory`.

use decision_contract::decision_training_job::{QuestionRole, TrainingExample};
use decision_contract::head_tuning_file::TunableHeadFamily;
use decision_contract::{OrderedJsonValue, parse_system_one_questions};
use model_serving::decision_models::adapter_training::mlx_buffer_cache_reclaim::reclaimed_pass_count;
use serde::Deserialize;
use test_harness::test_run::TestRunContext;

use super::whole_model_positive_control::coded_job;
use super::whole_model_training::{loaded_engine, train_to_completion};

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
    let outcome = train_to_completion(decision_engine.as_mut(), job, "check-set-training");
    let reclaimed_passes = reclaimed_pass_count() - reclaimed_passes_before;
    eprintln!(
        "[check-set-training] family=Laya parity_gap={} reclaimed_passes={reclaimed_passes} runs={:?}",
        outcome.largest_parity_gap, outcome.runs
    );
    assert_eq!(outcome.largest_parity_gap, 0.0);
    assert!(
        reclaimed_passes > 0,
        "no pass cleared the MLX buffer cache, so the journey no longer exercises the clear"
    );
}
