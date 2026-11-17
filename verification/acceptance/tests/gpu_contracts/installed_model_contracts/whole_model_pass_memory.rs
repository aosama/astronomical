//! How much memory one whole-model gradient pass holds, by the length of
//! its one text, on a real checkpoint.
//!
//! Training sizes its passes from a measured cost per padded token. A text
//! longer than a pass's budget still runs whole, so whether the cost grows
//! with the length, or faster, decides whether long texts fit. This runs
//! one gradient step on a single text at a few lengths, on one loaded
//! model, and prints the model's resident memory and each step's peak, so
//! the fixed, per-token and per-token-squared parts can be read off in
//! about two minutes instead of from a failed training. The model loads
//! under a memory limit whose pass budget holds every one of these texts,
//! so all of them keep every activation and the measurements compare like
//! with like; longer texts are recomputed layer by layer, which
//! `whole_model_layer_recomputation` measures.
//!
//! Run with `cargo test-clef-whole-model-pass-memory` and
//! `--configuration PATH` naming Clef's `model_directory`.

use astronomical_decision_contract::OrderedJsonValue;
use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingStage, QuestionRole, TrainingExample, TrainingScope,
};
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_model_serving::{
    ClefDecisionEngine, DecisionEngine, DecisionTrainingStep, KevDecisionEngine,
};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

/// Words of Romeo and Juliet per text; about 1.3 tokens a word, so roughly
/// 130, 195 and 260 tokens plus the family's prompt, inside the pass budget
/// under [`MEMORY_LIMIT_BYTES`]. Clef's prompt adds enough that 300 words
/// already crossed it and was recomputed (1.64 GB above the model instead
/// of about 8), which would hide the per-token slope.
const WORDS_PER_TEXT: [usize; 3] = [100, 150, 200];
/// A quarter of this over Clef's measured 17 MB a token is a pass budget of
/// about 570 tokens.
const MEMORY_LIMIT_BYTES: usize = 36 * 1024 * 1024 * 1024;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const QUESTIONS_JSON: &str = r#"{"mood": {"type": "choice", "instructions": "Which mood does this passage carry?", "criteria": ["calm", "stormy"]}}"#;
/// A gigabyte as Astronomical counts it, 2^30 bytes.
const BYTES_PER_GB: f64 = 1_073_741_824.0;

fn one_text_job(word_count: usize) -> DecisionTrainingJob {
    DecisionTrainingJob {
        examples: vec![TrainingExample {
            state: OrderedJsonValue::Text(romeo_and_juliet::passage(0, word_count)),
            questions: serde_json::from_str(QUESTIONS_JSON).expect("the question is well formed"),
            question_roles: vec![QuestionRole::Training { correct_option: 0 }],
        }],
        scope: TrainingScope::WholeModel,
    }
}

/// MLX's peak memory over the job's first gradient step, in bytes.
fn first_step_peak_bytes(decision_engine: &mut dyn DecisionEngine, word_count: usize) -> usize {
    decision_engine
        .start_training(one_text_job(word_count))
        .expect("the whole model should be trainable");
    let mut peak_was_reset = false;
    loop {
        let step = decision_engine
            .advance_training()
            .expect("every training step should succeed");
        let DecisionTrainingStep::Progress {
            stage, completed, ..
        } = step
        else {
            panic!("the training finished before its first step was measured");
        };
        if stage != DecisionTrainingStage::Training {
            continue;
        }
        if !peak_was_reset {
            mlx_bridge::reset_peak_memory().expect("the MLX peak resets");
            peak_was_reset = true;
        } else if completed >= 1 {
            let peak_bytes = mlx_bridge::peak_memory_bytes().expect("the MLX peak reads");
            decision_engine.cancel_training();
            return peak_bytes;
        }
    }
}

fn engine_under_probe_limit(
    family: TunableHeadFamily,
    model_directory: &std::path::Path,
) -> Box<dyn DecisionEngine> {
    let model_id = "whole-model-pass-memory";
    let mut decision_engine: Box<dyn DecisionEngine> = match family {
        TunableHeadFamily::Kev => Box::new(
            KevDecisionEngine::new(
                model_id,
                model_directory,
                MEMORY_LIMIT_BYTES,
                MLX_BUFFER_CACHE_LIMIT_BYTES,
            )
            .expect("the Kev folder should validate"),
        ),
        _ => Box::new(
            ClefDecisionEngine::new(
                model_id,
                model_directory,
                MEMORY_LIMIT_BYTES,
                MLX_BUFFER_CACHE_LIMIT_BYTES,
            )
            .expect("the Clef folder should validate"),
        ),
    };
    decision_engine.load().expect("the model should load");
    decision_engine
}

async fn report_pass_memory(family: TunableHeadFamily) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let mut decision_engine = engine_under_probe_limit(family, &model_directory);
    let resident_bytes = mlx_bridge::active_memory_bytes().expect("MLX active memory reads");
    eprintln!(
        "[whole-model-pass-memory] family={family:?} resident_gb={:.2}",
        resident_bytes as f64 / BYTES_PER_GB
    );
    for word_count in WORDS_PER_TEXT {
        let peak_bytes = first_step_peak_bytes(decision_engine.as_mut(), word_count);
        eprintln!(
            "[whole-model-pass-memory] family={family:?} words={word_count} step_peak_gb={:.2} above_resident_gb={:.2}",
            peak_bytes as f64 / BYTES_PER_GB,
            peak_bytes.saturating_sub(resident_bytes) as f64 / BYTES_PER_GB
        );
    }
}

#[ignore = "requires a Clef model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_clef_whole_model_pass_memory_by_text_length() {
    report_pass_memory(TunableHeadFamily::Clef).await;
}

#[ignore = "requires a Kev model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_kev_whole_model_pass_memory_by_text_length() {
    report_pass_memory(TunableHeadFamily::Kev).await;
}
