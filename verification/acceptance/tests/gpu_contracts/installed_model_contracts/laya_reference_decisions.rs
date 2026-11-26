//! Compares the Laya backend with the reference implementation on real checkpoints.
//!
//! The fixture (`tests/fixtures/laya_reference_decisions.json`) holds what the
//! reference package produced in float32 on the CPU from the original
//! `convaiinnovations/laya` checkpoints, at the revision the fixture records:
//! for every question, the token sequence the model read, its raw option
//! logits, the temperature applied and the resulting probabilities. This
//! journey loads an 8-bit MLX conversion of the same checkpoints and checks
//! two things:
//!
//! * the token sequences are identical, which proves the prompt layout on the
//!   real tokenizer, and
//! * the probabilities agree within a stated tolerance, which bounds what
//!   8-bit weights and a different runtime change.
//!
//! Run it with `cargo test-laya-reference-decisions --configuration PATH`,
//! where the configuration's `model_directory` is an MLX Laya folder: the
//! published `cavi-ai/laya-MLX-8bit` build, or the output of
//! `cargo xtask convert-laya`. Either way the English checkpoint sits at its
//! root and the multilingual one in `multilingual/`. The states are Shakespeare passages, in English and, for
//! the multilingual checkpoint, also in German and French.

use std::path::Path;

use decision_contract::{
    HeadInputRequest, OrderedJsonValue, QuestionOutcome, parse_system_one_questions,
};
use model_serving::{DecisionEngine, LayaDecisionEngine, laya_question_sequences};
use test_harness::test_run::TestRunContext;

use super::decision_parity_measurement::ParityMeasurement;
use super::decision_reference_fixture::{
    reference_array, reference_field, reference_number, reference_numbers,
};

/// The whole Laya repository is under 2 GB; this leaves ample forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 6 * 1024 * 1024 * 1024;
const TEST_MLX_BUFFER_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// The largest probability difference accepted against the float32 reference.
///
/// The compared weights are quantized to 8 bits, so some movement is expected.
/// Measured on an Apple Silicon Mac when this journey was added: 0.033 on the
/// English checkpoint and 0.016 on the multilingual one. The English maximum
/// is on a 12-option question, whose shipped temperature is clamped to 0.5 and
/// so doubles the logit noise; every other question stayed under 0.017. The
/// tolerance leaves room above those measurements for another machine's rounding.
const MAXIMUM_PROBABILITY_DIFFERENCE: f64 = 0.05;
const REFERENCE_DECISIONS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/laya_reference_decisions.json"
));

fn compare_question(
    case_name: &str,
    question_outcome: &QuestionOutcome,
    reference_question: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let reference_probabilities = reference_numbers(reference_question, "probabilities");
    let reference_logits = reference_numbers(reference_question, "option_logits");
    let reference_temperature =
        reference_number(reference_field(reference_question, "temperature"));
    assert!(
        (question_outcome.applied_temperature - reference_temperature).abs() < 1e-9,
        "case {case_name} question {}: temperature {} differs from the reference {reference_temperature}",
        question_outcome.question_name,
        question_outcome.applied_temperature
    );
    measurement.record_question(
        "laya",
        case_name,
        question_outcome,
        &reference_probabilities,
        &reference_logits,
        &format!(" temperature={:.3}", question_outcome.applied_temperature),
    );
}

fn compare_checkpoint(checkpoint_name: &str, checkpoint_directory: &Path) -> ParityMeasurement {
    let reference: OrderedJsonValue =
        serde_json::from_str(REFERENCE_DECISIONS_JSON).expect("the reference fixture should parse");
    let mut reference_checkpoint = None;
    for candidate_checkpoint in reference_array(&reference, "checkpoints") {
        if reference_field(candidate_checkpoint, "name").as_text() == Some(checkpoint_name) {
            reference_checkpoint = Some(candidate_checkpoint);
        }
    }
    let reference_checkpoint = reference_checkpoint
        .unwrap_or_else(|| panic!("the fixture has a `{checkpoint_name}` checkpoint"));
    let mut decision_engine = LayaDecisionEngine::new(
        checkpoint_name,
        checkpoint_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate");
    decision_engine
        .load()
        .expect("the Laya checkpoint should load");
    let mut measurement = ParityMeasurement::default();
    for reference_case in reference_array(reference_checkpoint, "cases") {
        compare_case(&mut decision_engine, reference_case, &mut measurement);
    }
    measurement.report(&format!("[laya-parity] checkpoint={checkpoint_name}"));
    measurement
}

fn compare_case(
    decision_engine: &mut LayaDecisionEngine,
    reference_case: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let case_name = reference_field(reference_case, "name")
        .as_text()
        .expect("a case name");
    let state = reference_field(reference_case, "state");
    let questions = parse_system_one_questions(reference_field(reference_case, "questions"))
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    let reference_questions = reference_array(reference_case, "reference");
    let question_sequences = laya_question_sequences(&decision_engine, state, &questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    for (question_sequence, reference_question) in
        question_sequences.iter().zip(reference_questions)
    {
        let mut token_ids = Vec::with_capacity(question_sequence.token_ids.len());
        for token_id in &question_sequence.token_ids {
            token_ids.push(f64::from(*token_id));
        }
        assert_eq!(
            token_ids,
            reference_numbers(reference_question, "token_ids"),
            "case {case_name}: the token sequence differs from the reference"
        );
        let mut marker_positions = Vec::new();
        for marker_position in &question_sequence.option_marker_positions {
            marker_positions.push(*marker_position as f64);
        }
        assert_eq!(
            marker_positions,
            reference_numbers(reference_question, "marker_positions"),
            "case {case_name}: the option markers differ from the reference"
        );
    }
    let decision_output = decision_engine
        .decide(state, &questions, HeadInputRequest::Omit)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    for (question_outcome, reference_question) in decision_output
        .question_outcomes
        .iter()
        .zip(reference_questions)
    {
        compare_question(case_name, question_outcome, reference_question, measurement);
    }
}

fn assert_within_tolerance(measurement: &ParityMeasurement) {
    measurement.assert_within_tolerance(MAXIMUM_PROBABILITY_DIFFERENCE);
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_match_the_reference_decisions_on_the_english_checkpoint() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let repository_directory = TestRunContext::current().required_model_directory();
    assert_within_tolerance(&compare_checkpoint("english", &repository_directory));
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_match_the_reference_decisions_on_the_multilingual_checkpoint() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let repository_directory = TestRunContext::current().required_model_directory();
    assert_within_tolerance(&compare_checkpoint(
        "multilingual",
        &repository_directory.join("multilingual"),
    ));
}
