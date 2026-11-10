//! Measures how long the Laya engine takes to answer a request.
//!
//! This is a measurement, not a threshold test: it prints the latency of
//! requests with 1, 5 and 10 questions for each checkpoint and asserts only
//! that every request was answered. Each question's sequence is filled to the
//! checkpoint's own token limit by a long Romeo and Juliet state, which is the
//! most work a question can be, and one warmup request runs before timing.
//!
//! Decision models are measured this way and not with the repository's usual
//! 10,000 tokens in and 500 out: they generate nothing, and a question reads
//! at most 512 or 1,024 tokens.
//!
//! Run it with `cargo measure-laya-decision-latency --configuration PATH`,
//! where the configuration's `model_directory` is the MLX Laya repository
//! folder. The alias builds optimized code.

use std::path::Path;
use std::time::Instant;

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, LayaDecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

#[path = "../../support/romeo_and_juliet.rs"]
#[allow(dead_code)]
mod romeo_and_juliet;

const TEST_MLX_MEMORY_CEILING_BYTES: usize = 6 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const QUESTION_COUNTS: [usize; 3] = [1, 5, 10];
/// Timed requests per question count; the median of these is reported.
const TIMED_REQUESTS_PER_QUESTION_COUNT: usize = 20;
/// Words of state; far more than either checkpoint's token limit holds.
const STATE_WORD_COUNT: usize = 2_000;

fn questions_of_count(question_count: usize) -> Vec<DecisionQuestion> {
    let subjects = [
        "death", "love", "a sword", "the sea", "a feud", "night", "a prince", "poison", "a letter",
        "a tomb",
    ];
    let mut question_members = Vec::with_capacity(question_count);
    for question_index in 0..question_count {
        question_members.push(format!(
            r#""q{question_index}": {{"type": "noul", "instructions": "Does the passage mention {}?"}}"#,
            subjects[question_index % subjects.len()]
        ));
    }
    let questions: OrderedJsonValue =
        serde_json::from_str(&format!("{{{}}}", question_members.join(",")))
            .expect("the measurement questions are well formed");
    parse_system_one_questions(&questions).expect("the measurement questions are valid")
}

fn measure_checkpoint(checkpoint_name: &str, checkpoint_directory: &Path) {
    let load_started_at = Instant::now();
    let mut decision_engine = LayaDecisionEngine::new(
        checkpoint_name,
        checkpoint_directory,
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the Laya artifact should validate");
    let load_result = decision_engine.load().expect("the checkpoint should load");
    let state = OrderedJsonValue::Text(romeo_and_juliet::excerpt(STATE_WORD_COUNT));
    decision_engine
        .decide(&state, &questions_of_count(1))
        .expect("the warmup request should be answered");
    eprintln!(
        "[laya-latency] checkpoint={checkpoint_name} tokens_per_question={} load_and_warmup_milliseconds={}",
        load_result.maximum_tokens_per_question,
        load_started_at.elapsed().as_millis()
    );
    for question_count in QUESTION_COUNTS {
        let questions = questions_of_count(question_count);
        let mut request_milliseconds = Vec::with_capacity(TIMED_REQUESTS_PER_QUESTION_COUNT);
        let mut input_token_count = 0;
        for _timed_request in 0..TIMED_REQUESTS_PER_QUESTION_COUNT {
            let request_started_at = Instant::now();
            let decision_output = decision_engine
                .decide(&state, &questions)
                .expect("the timed request should be answered");
            request_milliseconds.push(request_started_at.elapsed().as_secs_f64() * 1_000.0);
            assert_eq!(decision_output.question_outcomes.len(), question_count);
            input_token_count = decision_output.input_token_count;
        }
        request_milliseconds.sort_by(f64::total_cmp);
        let median_milliseconds = request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT / 2];
        eprintln!(
            "[laya-latency] checkpoint={checkpoint_name} questions={question_count} input_tokens={input_token_count} median_milliseconds={median_milliseconds:.1} fastest_milliseconds={:.1} slowest_milliseconds={:.1} milliseconds_per_question={:.1}",
            request_milliseconds[0],
            request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT - 1],
            median_milliseconds / question_count as f64,
        );
    }
}

#[ignore = "requires the MLX Laya repository model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_laya_decision_latency_at_the_token_limit() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let repository_directory = TestRunContext::current().required_model_directory();
    measure_checkpoint("english", &repository_directory);
    measure_checkpoint("multilingual", &repository_directory.join("multilingual"));
}
