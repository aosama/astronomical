//! Compares the Clef backend with Clef's reference scoring on the real build.
//!
//! The fixture (`tests/fixtures/clef_reference_decisions_flash_8bit.json`)
//! holds what `generate_clef_reference_decisions.py` produced in float32 on
//! the CPU from `mlx-community/clef-flash-8bit`, the build Astronomical
//! serves: transformers' Qwen3.5 text model given the build's weights,
//! dequantized exactly, and `JointSchemaHead` as `joint_schema_model.py`
//! defines it. For every case it records the sequence's tokens and, per
//! question, each option's identifier, raw logit and probability in the
//! order Clef scores them. This journey loads the same build and checks two
//! things:
//!
//! * the sequence's tokens are identical, which proves the prompt and schema
//!   layout on the real tokenizer, and
//! * the probabilities agree within a stated tolerance, which bounds what
//!   MLX's bfloat16 forward and a different runtime change on identical
//!   weights.
//!
//! Run it with `cargo test-clef-reference-decisions --configuration PATH`,
//! where the configuration's `model_directory` is the build's folder.

use astronomical_decision_contract::{
    DecisionQuestion, DecisionQuestionBody, OrderedJsonValue, QuestionOutcome,
    parse_system_one_questions,
};
use astronomical_model_serving::{ClefDecisionEngine, DecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

use super::decision_reference_fixture::{
    answer_probabilities, most_probable_index, reference_array, reference_field, reference_number,
    reference_numbers,
};

/// The 8-bit build is about 11 GB; a 16,384-token forward needs room beside it.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 24 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
/// The largest probability difference accepted against the float32 reference.
///
/// Both sides read identical weights; the backbone runs in bfloat16 in MLX
/// and in float32 in the reference, so its activations round differently.
/// The measured values are stated in the pull request that added this.
const MAXIMUM_PROBABILITY_DIFFERENCE: f64 = 0.05;
const REFERENCE_DECISIONS_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/clef_reference_decisions_flash_8bit.json"
));

/// What the comparison measured.
#[derive(Default)]
struct ParityMeasurement {
    question_count: usize,
    same_answer_count: usize,
    largest_probability_difference: f64,
    probability_difference_total: f64,
    probability_count: usize,
    largest_logit_difference: f64,
}

/// The identifiers of a question's options in the contract's canonical
/// order, as Clef names them: a choice's keys, `"0"` upward for a score, and
/// `false` then `true` for a noul.
fn canonical_option_ids(question: &DecisionQuestion) -> Vec<String> {
    match &question.body {
        DecisionQuestionBody::Choice { options } => {
            let mut option_ids = Vec::with_capacity(options.len());
            for option in options {
                option_ids.push(option.key.clone());
            }
            option_ids
        }
        DecisionQuestionBody::Score { levels } => {
            let mut option_ids = Vec::with_capacity(levels.len());
            for level_index in 0..levels.len() {
                option_ids.push(level_index.to_string());
            }
            option_ids
        }
        DecisionQuestionBody::Noul { .. } => vec!["false".to_owned(), "true".to_owned()],
    }
}

/// `reference_key`'s values (Clef's order) put in the canonical order.
fn canonical_reference_values(
    question: &DecisionQuestion,
    reference_row: &OrderedJsonValue,
    reference_key: &str,
) -> Vec<f64> {
    let scored_option_ids = reference_array(reference_row, "option_ids");
    let scored_values = reference_numbers(reference_row, reference_key);
    let mut canonical_values = Vec::new();
    for canonical_option_id in canonical_option_ids(question) {
        let mut found_value = None;
        for (scored_option_id, scored_value) in scored_option_ids.iter().zip(&scored_values) {
            if scored_option_id.as_text() == Some(canonical_option_id.as_str()) {
                found_value = Some(*scored_value);
            }
        }
        canonical_values.push(
            found_value
                .unwrap_or_else(|| panic!("the reference scored no option {canonical_option_id}")),
        );
    }
    canonical_values
}

fn compare_question(
    case_name: &str,
    question: &DecisionQuestion,
    question_outcome: &QuestionOutcome,
    reference_row: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let reference_probabilities =
        canonical_reference_values(question, reference_row, "probabilities");
    let reference_logits = canonical_reference_values(question, reference_row, "option_logits");
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
        "[clef-parity] case={case_name} question={} options={} largest_probability_difference={question_probability_difference:.5}",
        question_outcome.question_name,
        probabilities.len(),
    );
    measurement.question_count += 1;
    if most_probable_index(&probabilities) == most_probable_index(&reference_probabilities) {
        measurement.same_answer_count += 1;
    }
}

fn compare_case(
    decision_engine: &mut ClefDecisionEngine,
    reference_case: &OrderedJsonValue,
    measurement: &mut ParityMeasurement,
) {
    let case_name = reference_field(reference_case, "name")
        .as_text()
        .expect("a case name");
    let state = reference_field(reference_case, "state");
    let questions = parse_system_one_questions(reference_field(reference_case, "questions"))
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    let mut token_ids = Vec::new();
    for token_id in decision_engine
        .sequence_token_ids(state, &questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"))
    {
        token_ids.push(f64::from(token_id));
    }
    let mut reference_token_ids = Vec::new();
    for reference_token_id in reference_array(reference_case, "token_ids") {
        reference_token_ids.push(reference_number(reference_token_id));
    }
    assert_eq!(
        token_ids, reference_token_ids,
        "case {case_name}: the sequence differs from the reference"
    );
    let reference_rows = reference_array(reference_case, "reference");
    let decision_output = decision_engine
        .decide(state, &questions)
        .unwrap_or_else(|error| panic!("case {case_name}: {error}"));
    for ((question, question_outcome), reference_row) in questions
        .iter()
        .zip(&decision_output.question_outcomes)
        .zip(reference_rows)
    {
        compare_question(
            case_name,
            question,
            question_outcome,
            reference_row,
            measurement,
        );
    }
}

#[ignore = "requires the Clef build as model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_match_the_reference_decisions_within_the_stated_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let reference: OrderedJsonValue =
        serde_json::from_str(REFERENCE_DECISIONS_JSON).expect("the reference fixture should parse");
    let mut decision_engine = ClefDecisionEngine::new(
        "clef",
        &TestRunContext::current().required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the Clef build should validate");
    decision_engine.load().expect("the Clef build should load");
    let mut measurement = ParityMeasurement::default();
    for reference_case in reference_array(&reference, "cases") {
        compare_case(&mut decision_engine, reference_case, &mut measurement);
    }
    eprintln!(
        "[clef-parity] questions={} same_answer={} largest_probability_difference={:.5} mean_probability_difference={:.5} largest_logit_difference={:.5}",
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
