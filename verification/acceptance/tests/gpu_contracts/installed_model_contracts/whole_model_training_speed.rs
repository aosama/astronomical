//! How fast a whole-model training runs, per question, on a real checkpoint.
//!
//! A training's time is its gradient steps (every labelled question run
//! forwards and backwards through the whole model) and its scoring (each
//! question run as serving runs it). This measures both on a small job in
//! about twenty seconds, so a change to the training path is measured in
//! seconds rather than by a full training.
//!
//! Run one family with `cargo test-kev-whole-model-training-speed`,
//! `cargo test-clef-whole-model-training-speed` or
//! `cargo test-laya-whole-model-training-speed` and `--configuration PATH`
//! naming that family's `model_directory`.

use std::time::{Duration, Instant};

use decision_contract::OrderedJsonValue;
use decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingStage, QuestionRole, TrainingExample, TrainingScope,
};
use decision_contract::head_tuning_file::TunableHeadFamily;
use model_serving::DecisionTrainingStep;
use test_harness::test_run::TestRunContext;

use super::whole_model_training::loaded_engine;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

/// Of every eight examples seven train and one chooses: enough labelled
/// questions for several full steps of sixteen before the pass ends.
const EXAMPLE_COUNT: usize = 64;
const WORDS_PER_STATE: usize = 40;
/// Steps timed; the first is left out, as it builds what later steps reuse.
const TIMED_STEP_COUNT: u64 = 3;
/// Labelled questions in one optimizer step (`WHOLE_MODEL_TRAINING_SETTINGS`).
const QUESTIONS_PER_STEP: u64 = 16;
const QUESTIONS_JSON: &str = r#"{"mood": {"type": "choice", "instructions": "Which mood does this passage carry?", "criteria": ["calm", "stormy"]}}"#;

fn speed_job() -> DecisionTrainingJob {
    let mut examples = Vec::with_capacity(EXAMPLE_COUNT);
    for example_index in 0..EXAMPLE_COUNT {
        let correct_option = example_index % 2;
        let role = if example_index % 8 == 7 {
            QuestionRole::Choosing { correct_option }
        } else {
            QuestionRole::Training { correct_option }
        };
        examples.push(TrainingExample {
            state: OrderedJsonValue::Text(romeo_and_juliet::passage(
                example_index * WORDS_PER_STATE,
                WORDS_PER_STATE,
            )),
            questions: serde_json::from_str(QUESTIONS_JSON).expect("the question is well formed"),
            question_roles: vec![role],
        });
    }
    DecisionTrainingJob {
        examples,
        scope: TrainingScope::WholeModel,
    }
}

const BYTES_PER_GIB: f64 = 1_073_741_824.0;

/// What one measurement found.
struct TrainingSpeed {
    scoring_milliseconds_per_question: f64,
    gradient_milliseconds_per_question: f64,
    /// MLX's peak memory while stepping, the model included.
    stepping_peak_gib: f64,
}

/// Milliseconds per scored question while checking, and per labelled
/// question while stepping.
async fn measure_training_speed(family: TunableHeadFamily) -> TrainingSpeed {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let mut decision_engine = loaded_engine(family, &model_directory);
    decision_engine
        .start_training(speed_job())
        .expect("the whole model should be trainable");
    let mut checking_started = None;
    let mut checking_time = Duration::ZERO;
    let mut timing_from: Option<(Instant, u64)> = None;
    loop {
        let step = decision_engine
            .advance_training()
            .expect("every training step should succeed");
        let DecisionTrainingStep::Progress {
            stage, completed, ..
        } = step
        else {
            panic!("the training finished before enough steps were timed");
        };
        match stage {
            DecisionTrainingStage::Checking if checking_started.is_none() => {
                checking_started = Some(Instant::now());
            }
            DecisionTrainingStage::Training => {
                if let Some(started) = checking_started.take() {
                    checking_time = started.elapsed();
                }
                match timing_from {
                    None if completed >= 1 => {
                        mlx_bridge::reset_peak_memory().expect("the MLX peak resets");
                        timing_from = Some((Instant::now(), completed));
                    }
                    Some((started, first_step)) if completed >= first_step + TIMED_STEP_COUNT => {
                        let stepping_time = started.elapsed();
                        decision_engine.cancel_training();
                        let timed_questions =
                            ((completed - first_step) * QUESTIONS_PER_STEP) as f64;
                        return TrainingSpeed {
                            scoring_milliseconds_per_question: checking_time.as_secs_f64() * 1000.0
                                / EXAMPLE_COUNT as f64,
                            gradient_milliseconds_per_question: stepping_time.as_secs_f64()
                                * 1000.0
                                / timed_questions,
                            stepping_peak_gib: mlx_bridge::peak_memory_bytes()
                                .expect("the MLX peak reads")
                                as f64
                                / BYTES_PER_GIB,
                        };
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

async fn report_training_speed(family: TunableHeadFamily) {
    let speed = measure_training_speed(family).await;
    eprintln!(
        "[whole-model-speed] family={family:?} scoring_ms_per_question={:.1} gradient_ms_per_question={:.1} stepping_peak_gib={:.2}",
        speed.scoring_milliseconds_per_question,
        speed.gradient_milliseconds_per_question,
        speed.stepping_peak_gib
    );
}

#[ignore = "requires a Kev model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_kev_whole_model_training_speed() {
    report_training_speed(TunableHeadFamily::Kev).await;
}

#[ignore = "requires a Clef model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_clef_whole_model_training_speed() {
    report_training_speed(TunableHeadFamily::Clef).await;
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_laya_whole_model_training_speed() {
    report_training_speed(TunableHeadFamily::Laya).await;
}
