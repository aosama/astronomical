//! How the Laya engine handles requests on a real checkpoint.
//!
//! Agreement with the reference implementation is the sibling
//! `laya_reference_decisions` journey's job. This one covers the engine's own
//! request handling on the English checkpoint: which requests it refuses and
//! why, that a request too large for one forward pass gets the answers it
//! would get split up, and what it reports about cut input.
//!
//! Run it with `cargo test-laya-engine-requests --configuration PATH`, where
//! the configuration's `model_directory` is the MLX Laya repository folder.

use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, LayaDecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const TEST_MLX_MEMORY_CEILING_BYTES: usize = 6 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
/// The English checkpoint reads at most this many tokens per question.
const ENGLISH_MAXIMUM_TOKENS_PER_QUESTION: u32 = 512;
/// Enough questions that their padded sequences exceed one forward pass.
const QUESTIONS_SPANNING_SEVERAL_PASSES: usize = 40;
/// A question answered in a full batch and alone goes through differently
/// shaped kernels, so its probability agrees to rounding, not bit for bit.
const CROSS_PASS_PROBABILITY_TOLERANCE: f64 = 1e-3;

fn new_english_engine() -> LayaDecisionEngine {
    LayaDecisionEngine::new(
        "laya-english",
        &TestRunContext::current().required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate")
}

fn questions_from(questions_json_text: &str) -> Vec<DecisionQuestion> {
    let questions: OrderedJsonValue =
        serde_json::from_str(questions_json_text).expect("test questions are well formed");
    parse_system_one_questions(&questions).expect("the test questions are valid")
}

fn text_state(word_count: usize) -> OrderedJsonValue {
    OrderedJsonValue::Text(romeo_and_juliet::excerpt(word_count))
}

fn probability_true(answer: &DecisionAnswer) -> f64 {
    match answer {
        DecisionAnswer::Noul(noul_answer) => noul_answer.probability_true,
        other_answer => panic!("expected a noul answer, got {other_answer:?}"),
    }
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_refuse_to_decide_before_it_is_loaded() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = new_english_engine();
    let questions = questions_from(r#"{"q": {"type": "noul", "instructions": "Is it night?"}}"#);
    assert!(matches!(
        decision_engine.decide(&text_state(20), &questions),
        Err(DecisionFailure::FatalExecution { .. })
    ));
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_refuse_requests_the_checkpoint_cannot_answer_and_say_why() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = new_english_engine();
    let load_result = decision_engine.load().expect("the checkpoint should load");
    assert_eq!(
        load_result.maximum_tokens_per_question,
        ENGLISH_MAXIMUM_TOKENS_PER_QUESTION
    );
    assert!(matches!(
        decision_engine.load(),
        Err(DecisionFailure::FatalExecution { .. })
    ));

    let no_questions: Vec<DecisionQuestion> = Vec::new();
    assert!(matches!(
        decision_engine.decide(&text_state(20), &no_questions),
        Err(DecisionFailure::InvalidRequest { .. })
    ));

    let without_instructions = questions_from(r#"{"q": {"type": "noul"}}"#);
    let Err(DecisionFailure::InvalidRequest { reason }) =
        decision_engine.decide(&text_state(20), &without_instructions)
    else {
        panic!("a question without instructions must be refused as the request's fault");
    };
    assert!(
        reason.contains("instructions"),
        "the reason names the fix: {reason}"
    );

    // 200 options cannot all keep a marker inside 512 tokens.
    let mut option_labels = Vec::new();
    for option_index in 0..200 {
        option_labels.push(format!("\"the option numbered {option_index}\""));
    }
    let too_many_options = questions_from(&format!(
        r#"{{"q": {{"type": "choice", "instructions": "Pick one.", "criteria": [{}]}}}}"#,
        option_labels.join(",")
    ));
    let Err(DecisionFailure::InvalidRequest { reason }) =
        decision_engine.decide(&text_state(20), &too_many_options)
    else {
        panic!("options that do not fit must be refused as the request's fault");
    };
    assert!(
        reason.contains("fewer options"),
        "the reason names the fix: {reason}"
    );
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_answer_a_request_spanning_several_passes_as_it_answers_each_question_alone() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = new_english_engine();
    decision_engine.load().expect("the checkpoint should load");
    // The state fills every sequence to the limit, so 40 questions need
    // 40 * 512 = 20,480 padded tokens: more than one pass holds.
    let long_state = text_state(2_000);
    let subjects = [
        "death", "love", "a sword", "the sea", "a feud", "night", "a prince", "poison",
    ];
    let mut question_members = Vec::new();
    for question_index in 0..QUESTIONS_SPANNING_SEVERAL_PASSES {
        let subject = subjects[question_index % subjects.len()];
        question_members.push(format!(
            r#""q{question_index}": {{"type": "noul", "instructions": "Does passage {question_index} mention {subject}?"}}"#
        ));
    }
    let all_questions = questions_from(&format!("{{{}}}", question_members.join(",")));
    let whole_request = decision_engine
        .decide(&long_state, &all_questions)
        .expect("the large request should be answered");
    assert_eq!(
        whole_request.question_outcomes.len(),
        QUESTIONS_SPANNING_SEVERAL_PASSES
    );
    assert_eq!(
        whole_request.input_token_count,
        QUESTIONS_SPANNING_SEVERAL_PASSES as u32 * ENGLISH_MAXIMUM_TOKENS_PER_QUESTION
    );
    for question_index in [0, 17, QUESTIONS_SPANNING_SEVERAL_PASSES - 1] {
        let outcome_in_the_whole = &whole_request.question_outcomes[question_index];
        assert_eq!(
            outcome_in_the_whole.question_name,
            format!("q{question_index}")
        );
        let coverage = outcome_in_the_whole.input_coverage;
        assert!(
            coverage.state_tokens_read < coverage.state_token_count,
            "the long state must be reported as cut"
        );
        let alone = decision_engine
            .decide(&long_state, &all_questions[question_index..=question_index])
            .expect("one question should be answered");
        let probability_difference = (probability_true(&outcome_in_the_whole.answer)
            - probability_true(&alone.question_outcomes[0].answer))
        .abs();
        eprintln!(
            "[laya-requests] question={question_index} cross_pass_probability_difference={probability_difference:e}"
        );
        assert!(probability_difference <= CROSS_PASS_PROBABILITY_TOLERANCE);
    }
}
