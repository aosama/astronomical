//! Measures how long the Kev engine takes to answer a request.
//!
//! This is a measurement, not a threshold test: it prints the latency of
//! requests with 1, 5 and 10 questions and asserts only that every request
//! was answered. The state is filled with Romeo and Juliet to Kev's state
//! limit, 65,536 tokens with the state token, which is the most work a
//! request can be: the state is prefilled once per request and every
//! question continues it. One warmup request runs before timing.
//!
//! Decision models are measured this way and not with the repository's usual
//! 10,000 tokens in and 500 out: they generate nothing.
//!
//! Run it with `cargo measure-kev-decision-latency --configuration PATH`,
//! where the configuration's `model_directory` is a `cargo xtask
//! convert-kev` output folder. The alias builds optimized code.

use std::time::Instant;

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{DecisionEngine, KevDecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

/// Kev-9B in bfloat16 is about 18 GB; a 65,536-token prefill needs room beside it.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 36 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
const QUESTION_COUNTS: [usize; 3] = [1, 5, 10];
/// Timed requests per question count; the median is reported. Each one
/// prefills a full 65,536-token state, so a few already take minutes on 9B.
const TIMED_REQUESTS_PER_QUESTION_COUNT: usize = 3;
/// Kev's serving limit on a state, its state token included.
const KEV_MAXIMUM_STATE_TOKENS: usize = 65_536;
const PROGRESS_INTERVAL_SECONDS: f64 = 5.0;
const ROMEO_AND_JULIET_WORDS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/model_metrics_100000_romeo_and_juliet_words.txt"
));

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

fn state_of_words(word_count: usize) -> OrderedJsonValue {
    OrderedJsonValue::Text(
        ROMEO_AND_JULIET_WORDS
            .split_whitespace()
            .take(word_count)
            .collect::<Vec<_>>()
            .join(" "),
    )
}

/// The longest Romeo and Juliet state within Kev's state limit, found by
/// bisecting on whole words with the engine's own tokenizer.
fn state_at_the_limit(decision_engine: &KevDecisionEngine) -> (OrderedJsonValue, usize) {
    let one_question = questions_of_count(1);
    let mut fitting_word_count = 0;
    let mut overflowing_word_count = ROMEO_AND_JULIET_WORDS.split_whitespace().count() + 1;
    while overflowing_word_count - fitting_word_count > 1 {
        let probe_word_count = fitting_word_count.midpoint(overflowing_word_count);
        let fits = decision_engine
            .question_row_layouts(&state_of_words(probe_word_count), &one_question)
            .is_ok();
        if fits {
            fitting_word_count = probe_word_count;
        } else {
            overflowing_word_count = probe_word_count;
        }
    }
    let state = state_of_words(fitting_word_count);
    let state_token_count = decision_engine
        .question_row_layouts(&state, &one_question)
        .expect("the bisected state fits")[0]
        .state_token_count;
    (state, state_token_count)
}

fn timed_request_milliseconds(
    decision_engine: &mut KevDecisionEngine,
    state: &OrderedJsonValue,
    question_count: usize,
) -> (f64, u32) {
    let questions = questions_of_count(question_count);
    let request_started_at = Instant::now();
    let decision_output = decision_engine
        .decide(state, &questions)
        .expect("the timed request should be answered");
    assert_eq!(decision_output.question_outcomes.len(), question_count);
    (
        request_started_at.elapsed().as_secs_f64() * 1_000.0,
        decision_output.input_token_count,
    )
}

#[ignore = "requires a converted Kev folder as model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_kev_decision_latency_at_the_state_limit() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let load_started_at = Instant::now();
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
    let (state, state_token_count) = state_at_the_limit(&decision_engine);
    assert!(state_token_count > KEV_MAXIMUM_STATE_TOKENS - 64);
    timed_request_milliseconds(&mut decision_engine, &state, 1);
    eprintln!(
        "[kev-latency] state_tokens={state_token_count} load_and_warmup_milliseconds={}",
        load_started_at.elapsed().as_millis()
    );
    let request_total = QUESTION_COUNTS.len() * TIMED_REQUESTS_PER_QUESTION_COUNT;
    let measurement_started_at = Instant::now();
    let mut last_report_seconds = 0.0;
    let mut requests_done = 0;
    for question_count in QUESTION_COUNTS {
        let mut request_milliseconds = Vec::with_capacity(TIMED_REQUESTS_PER_QUESTION_COUNT);
        let mut input_token_count = 0;
        for _timed_request in 0..TIMED_REQUESTS_PER_QUESTION_COUNT {
            let (milliseconds, input_tokens) =
                timed_request_milliseconds(&mut decision_engine, &state, question_count);
            request_milliseconds.push(milliseconds);
            input_token_count = input_tokens;
            requests_done += 1;
            let elapsed_seconds = measurement_started_at.elapsed().as_secs_f64();
            if elapsed_seconds - last_report_seconds >= PROGRESS_INTERVAL_SECONDS {
                last_report_seconds = elapsed_seconds;
                eprintln!(
                    "[kev-latency] progress requests={requests_done}/{request_total} percent={:.0} eta_seconds={:.0}",
                    100.0 * requests_done as f64 / request_total as f64,
                    elapsed_seconds / requests_done as f64 * (request_total - requests_done) as f64
                );
            }
        }
        request_milliseconds.sort_by(f64::total_cmp);
        let median_milliseconds = request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT / 2];
        eprintln!(
            "[kev-latency] questions={question_count} input_tokens={input_token_count} median_milliseconds={median_milliseconds:.1} fastest_milliseconds={:.1} slowest_milliseconds={:.1} state_prefill_tokens_per_second={:.0}",
            request_milliseconds[0],
            request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT - 1],
            state_token_count as f64 / (median_milliseconds / 1_000.0),
        );
    }
}
