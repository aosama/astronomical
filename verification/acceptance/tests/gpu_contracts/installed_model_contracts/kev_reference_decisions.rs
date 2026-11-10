//! Compares the Kev backend with Kev's reference scoring on real checkpoints.
//!
//! The fixtures (`tests/fixtures/kev_reference_decisions_*.json`) hold what
//! `generate_kev_reference_decisions.py` produced in float32 on the CPU: the
//! original adapter applied beside the frozen Qwen3.5 base, as peft applies
//! it, and every question run as one causal row. For every question the
//! fixture records the row's token identifiers, the offsets the pointer head
//! reads, the raw option logits and the probabilities. This journey loads a
//! `cargo xtask convert-kev` folder, whose adapter is merged into bfloat16
//! weights, and checks two things:
//!
//! * every row's tokens and readout offsets are identical, which proves the
//!   rendering and layout on the real tokenizer, and
//! * the probabilities agree within a stated tolerance, which bounds what the
//!   merge into bfloat16 and a different runtime change.
//!
//! Run it with `cargo test-kev-reference-decisions --configuration PATH`,
//! where the configuration's `model_directory` is a converted Kev folder. The
//! fixture is chosen by the base model the folder's `kev_decision.json` names.

use std::path::Path;

use astronomical_decision_contract::{
    OrderedJsonValue, QuestionOutcome, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, KevDecisionEngine, KevRowLayout};
use astronomical_test_harness::test_run::TestRunContext;

use super::decision_reference_fixture::{
    answer_probabilities, most_probable_index, reference_array, reference_field, reference_numbers,
};

/// Kev-9B in bfloat16 is about 18 GB; this leaves its forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 32 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
/// The largest probability difference accepted against the float32 reference.
///
/// The converted backbone holds bfloat16 weights with the adapter merged in,
/// while the reference keeps the adapter in float32 beside the frozen base,
/// so some movement is expected; the measured values are stated in the pull
/// request that added each fixture.
const MAXIMUM_PROBABILITY_DIFFERENCE: f64 = 0.05;
/// Each fixture, by the base model its adapter was trained on.
const REFERENCE_FIXTURES: [(&str, &str); 2] = [
    (
        "Qwen/Qwen3.5-0.8B-Base",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/kev_reference_decisions_0_8b.json"
        )),
    ),
    (
        "Qwen/Qwen3.5-9B-Base",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/kev_reference_decisions_9b.json"
        )),
    ),
];

/// What one checkpoint's comparison measured.
#[derive(Default)]
struct ParityMeasurement {
    question_count: usize,
    same_answer_count: usize,
    largest_probability_difference: f64,
    probability_difference_total: f64,
    probability_count: usize,
    largest_logit_difference: f64,
}

/// The fixture for the converted folder's base model.
fn reference_for(model_directory: &Path) -> OrderedJsonValue {
    let kev_settings: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model_directory.join("kev_decision.json"))
            .expect("a converted Kev folder has kev_decision.json"),
    )
    .expect("kev_decision.json is JSON");
    let base_model = kev_settings["base_model"]
        .as_str()
        .expect("kev_decision.json names its base model");
    for (fixture_base_model, fixture_json) in REFERENCE_FIXTURES {
        if fixture_base_model == base_model {
            return serde_json::from_str(fixture_json).expect("the reference fixture should parse");
        }
    }
    panic!("no reference fixture describes a Kev trained on {base_model}");
}

fn assert_same_layout(
    case_name: &str,
    row_layout: &KevRowLayout,
    reference_row: &OrderedJsonValue,
) {
    let mut token_ids = Vec::with_capacity(row_layout.token_ids.len());
    for token_id in &row_layout.token_ids {
        token_ids.push(f64::from(*token_id));
    }
    assert_eq!(
        token_ids,
        reference_numbers(reference_row, "token_ids"),
        "case {case_name}: the row's tokens differ from the reference"
    );
    let mut option_closing_offsets = Vec::new();
    for option_closing_offset in &row_layout.option_closing_offsets {
        option_closing_offsets.push(*option_closing_offset as f64);
    }
    assert_eq!(
        option_closing_offsets,
        reference_numbers(reference_row, "option_closing_offsets"),
        "case {case_name}: the readout offsets differ from the reference"
    );
}

fn compare_question(
    case_name: &str,
    question_outcome: &QuestionOutcome,
    reference_row: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let reference_probabilities = reference_numbers(reference_row, "probabilities");
    let reference_logits = reference_numbers(reference_row, "option_logits");
    let probabilities = answer_probabilities(&question_outcome.answer);
    assert_eq!(probabilities.len(), reference_probabilities.len());
    let mut question_probability_difference = 0.0_f64;
    for (probability, reference_probability) in probabilities.iter().zip(&reference_probabilities) {
        let probability_difference = (probability - reference_probability).abs();
        question_probability_difference =
            question_probability_difference.max(probability_difference);
        measurement.probability_difference_total += probability_difference;
        measurement.probability_count += 1;
    }
    measurement.largest_probability_difference = measurement
        .largest_probability_difference
        .max(question_probability_difference);
    for (option_logit, reference_logit) in
        question_outcome.option_logits.iter().zip(&reference_logits)
    {
        measurement.largest_logit_difference = measurement
            .largest_logit_difference
            .max((f64::from(*option_logit) - reference_logit).abs());
    }
    eprintln!(
        "[kev-parity] case={case_name} question={} options={} largest_probability_difference={question_probability_difference:.5}",
        question_outcome.question_name,
        probabilities.len(),
    );
    measurement.question_count += 1;
    if most_probable_index(&probabilities) == most_probable_index(&reference_probabilities) {
        measurement.same_answer_count += 1;
    }
}

fn compare_case(
    decision_engine: &mut KevDecisionEngine,
    reference_case: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let case_name = reference_field(reference_case, "name")
        .as_text()
        .expect("a case name");
    let state = reference_field(reference_case, "state");
    let questions = parse_system_one_questions(reference_field(reference_case, "questions"))
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    let reference_rows = reference_array(reference_case, "reference");
    assert_eq!(questions.len(), reference_rows.len());
    let row_layouts = decision_engine
        .question_row_layouts(state, &questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    for (row_layout, reference_row) in row_layouts.iter().zip(reference_rows) {
        assert_same_layout(case_name, row_layout, reference_row);
    }
    let decision_output = decision_engine
        .decide(state, &questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    for (question_outcome, reference_row) in
        decision_output.question_outcomes.iter().zip(reference_rows)
    {
        compare_question(case_name, question_outcome, reference_row, measurement);
    }
}

#[ignore = "requires a converted Kev folder as model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_match_the_reference_decisions_within_the_stated_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = TestRunContext::current().required_model_directory();
    let reference = reference_for(&model_directory);
    let mut decision_engine = KevDecisionEngine::new(
        "kev",
        &model_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the converted Kev folder should validate");
    decision_engine
        .load()
        .expect("the converted Kev folder should load");
    let mut measurement = ParityMeasurement::default();
    for reference_case in reference_array(&reference, "cases") {
        compare_case(&mut decision_engine, reference_case, &mut measurement);
    }
    eprintln!(
        "[kev-parity] checkpoint={} questions={} same_answer={} largest_probability_difference={:.5} mean_probability_difference={:.5} largest_logit_difference={:.5}",
        reference_field(&reference, "checkpoint")
            .as_text()
            .unwrap_or("?"),
        measurement.question_count,
        measurement.same_answer_count,
        measurement.largest_probability_difference,
        measurement.probability_difference_total / measurement.probability_count as f64,
        measurement.largest_logit_difference,
    );
    assert!(
        measurement.question_count > 0,
        "the fixture compared no questions"
    );
    assert!(
        measurement.largest_probability_difference <= MAXIMUM_PROBABILITY_DIFFERENCE,
        "the largest probability difference {:.5} exceeds {MAXIMUM_PROBABILITY_DIFFERENCE}",
        measurement.largest_probability_difference
    );
}
