//! How the Kev engine handles requests on a real converted checkpoint.
//!
//! Agreement with Kev's reference scoring is `kev_reference_decisions`'s
//! job. This one checks what can be checked without the reference: that an
//! unambiguous question is answered the way anyone would answer it, that a
//! question reads exactly the same alone and among others (each continues
//! the shared state from a restored checkpoint), and that the engine refuses
//! what the reference refuses.
//!
//! Run it with `cargo test-kev-engine-requests --configuration PATH`, where
//! the configuration's `model_directory` is a `cargo xtask convert-kev`
//! output folder.

use astronomical_decision_contract::{
    DecisionAnswer, DecisionFailure, DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, KevDecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

/// Kev-9B in bfloat16 is about 18 GB; this leaves its forward workspace.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 32 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
/// Kev's serving limit on a state, its state token included.
const KEV_MAXIMUM_STATE_TOKENS: usize = 65_536;

const PROLOGUE: &str = "Two households, both alike in dignity, In fair Verona, where we lay our scene, \
From ancient grudge break to new mutiny, Where civil blood makes civil hands unclean.";

fn loaded_engine() -> KevDecisionEngine {
    let mut decision_engine = KevDecisionEngine::new(
        "kev",
        &TestRunContext::current().required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the converted Kev folder should validate");
    decision_engine
        .load()
        .expect("the converted Kev folder should load");
    decision_engine
}

fn questions_from(questions_json_text: &str) -> Vec<DecisionQuestion> {
    let questions: OrderedJsonValue =
        serde_json::from_str(questions_json_text).expect("test questions are well formed");
    parse_system_one_questions(&questions).expect("the test questions are valid")
}

fn answer_probabilities(answer: &DecisionAnswer) -> Vec<f64> {
    match answer {
        DecisionAnswer::Choice(choice_answer) => choice_answer
            .option_probabilities
            .iter()
            .map(|option_probability| option_probability.probability)
            .collect(),
        DecisionAnswer::Score(score_answer) => score_answer.level_probabilities.clone(),
        DecisionAnswer::Noul(noul_answer) => {
            vec![
                1.0 - noul_answer.probability_true,
                noul_answer.probability_true,
            ]
        }
    }
}

const THREE_QUESTIONS: &str = r#"{
    "subject": {"type": "choice", "instructions": "What is this passage about?",
        "criteria": {"feud": "two families in conflict", "cooking": "preparing food", "sailing": "a voyage at sea"}},
    "violence": {"type": "noul", "instructions": "Does the passage mention bloodshed?"},
    "tone": {"type": "score", "instructions": "How cheerful is the passage?",
        "criteria": ["grim", "neutral", "cheerful"]}
}"#;

#[ignore = "requires a convert-kev output model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_answer_an_unambiguous_question_the_way_anyone_would() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = loaded_engine();

    let engine_output = decision_engine
        .decide(
            &OrderedJsonValue::Text(PROLOGUE.to_owned()),
            &questions_from(THREE_QUESTIONS),
        )
        .expect("the request should be answered");

    for question_outcome in &engine_output.question_outcomes {
        let probabilities = answer_probabilities(&question_outcome.answer);
        eprintln!(
            "[kev-engine] question={} probabilities={probabilities:?} logits={:?}",
            question_outcome.question_name, question_outcome.option_logits
        );
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    let DecisionAnswer::Choice(subject_answer) = &engine_output.question_outcomes[0].answer else {
        panic!("a choice question answers with a choice");
    };
    assert_eq!(subject_answer.chosen_key, "feud");
    let state_coverage = engine_output.question_outcomes[0].input_coverage;
    assert!(!state_coverage.state_was_cut());
    assert_eq!(state_coverage.distinguishable_option_count, 3);
}

#[ignore = "requires a convert-kev output model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_answer_a_question_alone_exactly_as_among_others() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = loaded_engine();
    let state = OrderedJsonValue::Text(romeo_and_juliet::excerpt(150));
    let all_questions = questions_from(THREE_QUESTIONS);

    let answered_together = decision_engine
        .decide(&state, &all_questions)
        .expect("the request should be answered");
    for (question_index, question) in all_questions.iter().enumerate() {
        let answered_alone = decision_engine
            .decide(&state, std::slice::from_ref(question))
            .expect("the single question should be answered");

        assert_eq!(
            answered_alone.question_outcomes[0].option_logits,
            answered_together.question_outcomes[question_index].option_logits,
            "question {} must not see any other question",
            question.name
        );
    }
}

#[ignore = "requires a convert-kev output model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_refuse_a_state_over_the_serving_limit_and_say_how_long_it_is() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = loaded_engine();
    let long_state = romeo_and_juliet::excerpt(usize::MAX).repeat(6);

    let decide_outcome = decision_engine.decide(
        &OrderedJsonValue::Text(long_state),
        &questions_from(r#"{"q": {"type": "noul", "instructions": "Is it night?"}}"#),
    );

    let Err(DecisionFailure::InvalidRequest { reason }) = decide_outcome else {
        panic!(
            "a state over {KEV_MAXIMUM_STATE_TOKENS} tokens must be refused, got {decide_outcome:?}"
        );
    };
    assert!(reason.contains("65536-token limit"), "{reason}");
}

#[ignore = "requires a convert-kev output model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_refuse_to_decide_before_it_is_loaded() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let mut decision_engine = KevDecisionEngine::new(
        "kev",
        &TestRunContext::current().required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the converted Kev folder should validate");

    assert!(matches!(
        decision_engine.decide(
            &OrderedJsonValue::Text(PROLOGUE.to_owned()),
            &questions_from(r#"{"q": {"type": "noul", "instructions": "Is it night?"}}"#),
        ),
        Err(DecisionFailure::FatalExecution { .. })
    ));
}
