//! A text too long for a training pass's memory budget trains to the same
//! weights when its layers are recomputed, in less memory (issue #2299).
//!
//! The same one-text job trains twice on a real checkpoint: once under a
//! memory limit whose pass budget is shorter than the text, so every layer
//! is recomputed during the backward pass, and once under a limit large
//! enough to keep every activation. Recomputation runs the same operations
//! again, so the trained weights must agree to rounding, and the recomputed
//! run must peak lower.
//!
//! Run with `cargo test-kev-whole-model-layer-recomputation` (or the Clef
//! alias) and `--configuration PATH` naming that family's
//! `model_directory`. About a minute on Kev 0.8B.

use astronomical_decision_contract::OrderedJsonValue;
use astronomical_decision_contract::decision_training_job::{
    DecisionTrainingJob, DecisionTrainingOutcome, QuestionRole, TrainingExample, TrainingScope,
};
use astronomical_decision_contract::head_tuning_file::TunableHeadFamily;
use astronomical_model_serving::{
    ClefDecisionEngine, DecisionEngine, DecisionTrainingStep, KevDecisionEngine,
};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const GIB: usize = 1024 * 1024 * 1024;
const MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// Words of Romeo and Juliet in the one text: with each family's prompt
/// around it, over the recomputation threshold under the low limit and
/// within it under the high one (`recomputation_threshold_tokens`: half the
/// limit over the measured cost per token). Clef's prompt adds its own
/// instructions, so its text is shorter.
fn words_per_text(family: TunableHeadFamily) -> usize {
    match family {
        TunableHeadFamily::Kev => 600,
        _ => 300,
    }
}
const QUESTIONS_JSON: &str = r#"{"mood": {"type": "choice", "instructions": "Which mood does this passage carry?", "criteria": ["calm", "stormy"]}}"#;
/// Recomputation repeats the same operations; anything larger than rounding
/// in a weight means it computed something else.
const WEIGHT_TOLERANCE: f32 = 1e-4;

/// Memory limits that put the text over and under each family's
/// recomputation threshold.
fn recomputing_and_storing_limits(family: TunableHeadFamily) -> (usize, usize) {
    match family {
        // 3 MB a token: 4 GiB recomputes above about 715 tokens and still
        // holds the bf16 0.8B; 20 GiB above about 3,580.
        TunableHeadFamily::Kev => (4 * GIB, 20 * GIB),
        // 17 MB a token: 12 GiB recomputes above about 380 tokens and still
        // holds the 8-bit 9B; 36 GiB above about 1,140.
        _ => (12 * GIB, 36 * GIB),
    }
}

fn one_long_text_job(family: TunableHeadFamily) -> DecisionTrainingJob {
    DecisionTrainingJob {
        examples: vec![TrainingExample {
            state: OrderedJsonValue::Text(romeo_and_juliet::passage(0, words_per_text(family))),
            questions: serde_json::from_str(QUESTIONS_JSON).expect("the question is well formed"),
            question_roles: vec![QuestionRole::Training { correct_option: 1 }],
        }],
        scope: TrainingScope::WholeModel,
    }
}

fn engine_under(family: TunableHeadFamily, memory_limit_bytes: usize) -> Box<dyn DecisionEngine> {
    let model_directory = TestRunContext::current().required_model_directory();
    let model_id = "layer-recomputation";
    let mut decision_engine: Box<dyn DecisionEngine> = match family {
        TunableHeadFamily::Kev => Box::new(
            KevDecisionEngine::new(
                model_id,
                &model_directory,
                memory_limit_bytes,
                MLX_BUFFER_CACHE_LIMIT_BYTES,
            )
            .expect("the Kev folder should validate"),
        ),
        _ => Box::new(
            ClefDecisionEngine::new(
                model_id,
                &model_directory,
                memory_limit_bytes,
                MLX_BUFFER_CACHE_LIMIT_BYTES,
            )
            .expect("the Clef folder should validate"),
        ),
    };
    decision_engine.load().expect("the model should load");
    decision_engine
}

/// The finished training and MLX's peak memory while it trained.
fn trained_with_peak(
    family: TunableHeadFamily,
    memory_limit_bytes: usize,
) -> (DecisionTrainingOutcome, usize) {
    let mut decision_engine = engine_under(family, memory_limit_bytes);
    decision_engine
        .start_training(one_long_text_job(family))
        .expect("the whole model should be trainable");
    mlx_bridge::reset_peak_memory().expect("the MLX peak resets");
    loop {
        match decision_engine
            .advance_training()
            .expect("every training step should succeed")
        {
            DecisionTrainingStep::Progress { .. } => {}
            DecisionTrainingStep::Completed(outcome) => {
                let peak_bytes = mlx_bridge::peak_memory_bytes().expect("the MLX peak reads");
                return (*outcome, peak_bytes);
            }
        }
    }
}

/// The largest difference between two trainings' weights, add-ons included,
/// after checking the two hold the same tensors, all of them finite: a
/// comparison of nothing, or of NaN, must not pass.
fn largest_weight_difference(
    first: &DecisionTrainingOutcome,
    second: &DecisionTrainingOutcome,
) -> f32 {
    assert!(
        !first.adapter_tensors.is_empty(),
        "the training wrote no add-ons"
    );
    assert_eq!(first.trained_tensors.len(), second.trained_tensors.len());
    assert_eq!(first.adapter_tensors.len(), second.adapter_tensors.len());
    let mut largest_difference = 0.0_f32;
    let first_tensors = first.trained_tensors.iter().chain(&first.adapter_tensors);
    let second_tensors = second.trained_tensors.iter().chain(&second.adapter_tensors);
    for (first_tensor, second_tensor) in first_tensors.zip(second_tensors) {
        assert_eq!(first_tensor.name, second_tensor.name);
        assert_eq!(first_tensor.values.0.len(), second_tensor.values.0.len());
        for (first_value, second_value) in first_tensor.values.0.iter().zip(&second_tensor.values.0)
        {
            assert!(first_value.is_finite() && second_value.is_finite());
            largest_difference = largest_difference.max((first_value - second_value).abs());
        }
    }
    largest_difference
}

async fn check_layer_recomputation(family: TunableHeadFamily) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let (recomputing_limit, storing_limit) = recomputing_and_storing_limits(family);
    let (recomputed, recomputed_peak) = trained_with_peak(family, recomputing_limit);
    let (stored, stored_peak) = trained_with_peak(family, storing_limit);
    let largest_difference = largest_weight_difference(&recomputed, &stored);
    eprintln!(
        "[layer-recomputation] family={family:?} recomputed_texts={} stored_texts={} largest_weight_difference={largest_difference:.2e} recomputed_peak_gb={:.2} stored_peak_gb={:.2}",
        recomputed.recomputed_text_count,
        stored.recomputed_text_count,
        recomputed_peak as f64 / GIB as f64,
        stored_peak as f64 / GIB as f64,
    );
    assert!(
        recomputed.recomputed_text_count > 0,
        "the long text was not recomputed"
    );
    assert_eq!(stored.recomputed_text_count, 0, "the stored run recomputed");
    assert_eq!(
        recomputed.trained_tensors.len(),
        stored.trained_tensors.len()
    );
    assert!(largest_difference <= WEIGHT_TOLERANCE);
    assert!(recomputed_peak < stored_peak);
}

#[ignore = "requires a Kev model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_train_a_text_over_the_pass_budget_to_the_same_weights_in_less_memory_on_kev() {
    check_layer_recomputation(TunableHeadFamily::Kev).await;
}

#[ignore = "requires a Clef model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_train_a_text_over_the_pass_budget_to_the_same_weights_in_less_memory_on_clef() {
    check_layer_recomputation(TunableHeadFamily::Clef).await;
}
