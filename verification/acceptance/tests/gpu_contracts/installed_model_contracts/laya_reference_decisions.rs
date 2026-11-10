//! Compares the Laya backend with the reference implementation on real checkpoints.
//!
//! The fixture (`tests/fixtures/laya_reference_decisions.json`) holds what the
//! reference package produced in float32 on the CPU from the original
//! checkpoints: for every question, the token sequence the model read, its raw
//! option logits, the temperature applied and the resulting probabilities.
//! This journey loads the 8-bit MLX conversion of the same checkpoints and
//! checks two things:
//!
//! * the token sequences are identical, which proves the prompt layout on the
//!   real tokenizer, and
//! * the probabilities agree within a stated tolerance, which bounds what
//!   8-bit weights and a different runtime change.
//!
//! Run it with `cargo test-laya-reference-decisions --configuration PATH`,
//! where the configuration's `model_directory` is the MLX Laya repository
//! folder: the English checkpoint at its root, the multilingual one in
//! `multilingual/`. The states are Shakespeare passages, in English and, for
//! the multilingual checkpoint, also in German and French.

use std::path::Path;

use astronomical_decision_contract::{
    DecisionAnswer, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, LayaDecisionEngine, QuestionOutcome};
use astronomical_test_harness::test_run::TestRunContext;

/// The whole Laya repository is under 2 GB; this leaves ample forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 6 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
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

fn reference_field<'a>(reference: &'a OrderedJsonValue, key: &str) -> &'a OrderedJsonValue {
    reference
        .member(key)
        .unwrap_or_else(|| panic!("the reference fixture has a `{key}` field"))
}

fn reference_array<'a>(reference: &'a OrderedJsonValue, key: &str) -> &'a [OrderedJsonValue] {
    reference_field(reference, key)
        .as_array()
        .unwrap_or_else(|| panic!("reference field `{key}` is an array"))
}

fn reference_number(value: &OrderedJsonValue) -> f64 {
    match value {
        OrderedJsonValue::SignedInteger(integer) => *integer as f64,
        OrderedJsonValue::UnsignedInteger(integer) => *integer as f64,
        OrderedJsonValue::Float(float) => *float,
        other_value => panic!("expected a number, got {}", other_value.type_name()),
    }
}

fn reference_numbers(reference: &OrderedJsonValue, key: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    for value in reference_array(reference, key) {
        numbers.push(reference_number(value));
    }
    numbers
}

/// The probabilities of an answer in canonical option order.
fn answer_probabilities(answer: &DecisionAnswer) -> Vec<f64> {
    match answer {
        DecisionAnswer::Choice(choice_answer) => {
            let mut probabilities = Vec::new();
            for option_probability in &choice_answer.option_probabilities {
                probabilities.push(option_probability.probability);
            }
            probabilities
        }
        DecisionAnswer::Score(score_answer) => score_answer.level_probabilities.clone(),
        DecisionAnswer::Noul(noul_answer) => {
            vec![
                1.0 - noul_answer.probability_true,
                noul_answer.probability_true,
            ]
        }
    }
}

fn most_probable_index(probabilities: &[f64]) -> usize {
    let mut most_probable_index = 0;
    for (option_index, probability) in probabilities.iter().enumerate() {
        if *probability > probabilities[most_probable_index] {
            most_probable_index = option_index;
        }
    }
    most_probable_index
}

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
    let probabilities = answer_probabilities(&question_outcome.answer);
    assert_eq!(probabilities.len(), reference_probabilities.len());
    for (probability, reference_probability) in probabilities.iter().zip(&reference_probabilities) {
        let probability_difference = (probability - reference_probability).abs();
        measurement.largest_probability_difference = measurement
            .largest_probability_difference
            .max(probability_difference);
        measurement.probability_difference_total += probability_difference;
        measurement.probability_count += 1;
    }
    for (option_logit, reference_logit) in
        question_outcome.option_logits.iter().zip(&reference_logits)
    {
        measurement.largest_logit_difference = measurement
            .largest_logit_difference
            .max((f64::from(*option_logit) - reference_logit).abs());
    }
    let mut question_probability_difference = 0.0_f64;
    for (probability, reference_probability) in probabilities.iter().zip(&reference_probabilities) {
        question_probability_difference =
            question_probability_difference.max((probability - reference_probability).abs());
    }
    eprintln!(
        "[laya-parity] case={case_name} question={} options={} temperature={:.3} largest_probability_difference={question_probability_difference:.5}",
        question_outcome.question_name,
        probabilities.len(),
        question_outcome.applied_temperature,
    );
    measurement.question_count += 1;
    if most_probable_index(&probabilities) == most_probable_index(&reference_probabilities) {
        measurement.same_answer_count += 1;
    }
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
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate");
    decision_engine
        .load()
        .expect("the Laya checkpoint should load");
    let mut measurement = ParityMeasurement::default();
    for reference_case in reference_array(reference_checkpoint, "cases") {
        compare_case(&mut decision_engine, reference_case, &mut measurement);
    }
    eprintln!(
        "[laya-parity] checkpoint={checkpoint_name} questions={} same_answer={} largest_probability_difference={:.5} mean_probability_difference={:.5} largest_logit_difference={:.5}",
        measurement.question_count,
        measurement.same_answer_count,
        measurement.largest_probability_difference,
        measurement.probability_difference_total / measurement.probability_count as f64,
        measurement.largest_logit_difference,
    );
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
    let question_sequences = decision_engine
        .question_sequences(state, &questions)
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
        .decide(state, &questions)
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
    assert!(
        measurement.largest_probability_difference <= MAXIMUM_PROBABILITY_DIFFERENCE,
        "the largest probability difference {:.5} exceeds {MAXIMUM_PROBABILITY_DIFFERENCE}",
        measurement.largest_probability_difference
    );
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
