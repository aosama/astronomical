//! Measures how long the Clef engine takes to answer a request.
//!
//! This is a measurement, not a threshold test: it prints the latency of
//! requests with 1, 5 and 10 questions and asserts only that every request
//! was answered. Clef reads one sequence of at most 16,384 tokens holding
//! the state and every question, and cuts the state to fit, so a long Romeo
//! and Juliet state fills every request to the limit, which is the most work
//! a request can be. One warmup request runs before timing. Beside each
//! timing it prints the peak MLX memory the process has needed so far; the
//! value after the last request is what the catalog entry records as
//! `approximate_memory_bytes`.
//!
//! Decision models are measured this way and not with the repository's usual
//! 10,000 tokens in and 500 out: they generate nothing.
//!
//! Run it with `cargo measure-clef-decision-latency --configuration PATH`,
//! where the configuration's `model_directory` is the Clef build's folder.
//! The alias builds optimized code.

use std::time::Instant;

use astronomical_decision_contract::{
    DecisionQuestion, OrderedJsonValue, parse_system_one_questions,
};
use astronomical_model_serving::{ClefDecisionEngine, DecisionEngine};
use astronomical_test_harness::test_run::TestRunContext;

use super::decision_memory_peak::{peak_memory_bytes_after_request, peak_memory_report};

/// The 8-bit build is about 11 GB; a 16,384-token forward needs room beside it.
const TEST_MLX_MEMORY_CEILING_BYTES: usize = 24 * 1024 * 1024 * 1024;
const TEST_ALLOCATOR_CACHE_LIMIT_BYTES: usize = 1024 * 1024 * 1024;
const QUESTION_COUNTS: [usize; 3] = [1, 5, 10];
/// Timed requests per question count; the median is reported.
const TIMED_REQUESTS_PER_QUESTION_COUNT: usize = 3;
/// Clef's limit on the whole sequence.
const CLEF_MAXIMUM_SEQUENCE_TOKENS: u32 = 16_384;
/// Words of state: far more than the sequence holds, so it is always cut to fit.
const STATE_WORD_COUNT: usize = 20_000;
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

fn timed_request_milliseconds(
    decision_engine: &mut ClefDecisionEngine,
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

#[ignore = "requires the Clef build as model_directory in the explicit test-run configuration"]
#[tokio::test]
#[serial_test::serial]
async fn should_measure_clef_decision_latency_at_the_sequence_limit() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let load_started_at = Instant::now();
    let mut decision_engine = ClefDecisionEngine::new(
        "clef",
        &TestRunContext::current().required_model_directory(),
        TEST_MLX_MEMORY_CEILING_BYTES,
        TEST_ALLOCATOR_CACHE_LIMIT_BYTES,
    )
    .expect("the Clef build should validate");
    decision_engine.load().expect("the Clef build should load");
    let state = OrderedJsonValue::Text(
        ROMEO_AND_JULIET_WORDS
            .split_whitespace()
            .take(STATE_WORD_COUNT)
            .collect::<Vec<_>>()
            .join(" "),
    );
    let (_, warmup_input_tokens) = timed_request_milliseconds(&mut decision_engine, &state, 1);
    assert_eq!(warmup_input_tokens, CLEF_MAXIMUM_SEQUENCE_TOKENS);
    eprintln!(
        "[clef-latency] sequence_tokens={warmup_input_tokens} load_and_warmup_milliseconds={} {}",
        load_started_at.elapsed().as_millis(),
        peak_memory_report(peak_memory_bytes_after_request(&mut decision_engine)),
    );
    let request_total = QUESTION_COUNTS.len() * TIMED_REQUESTS_PER_QUESTION_COUNT;
    let measurement_started_at = Instant::now();
    let mut last_report_seconds = 0.0;
    let mut requests_done = 0;
    for question_count in QUESTION_COUNTS {
        let mut request_milliseconds = Vec::with_capacity(TIMED_REQUESTS_PER_QUESTION_COUNT);
        for _timed_request in 0..TIMED_REQUESTS_PER_QUESTION_COUNT {
            let (milliseconds, input_tokens) =
                timed_request_milliseconds(&mut decision_engine, &state, question_count);
            assert_eq!(input_tokens, CLEF_MAXIMUM_SEQUENCE_TOKENS);
            request_milliseconds.push(milliseconds);
            requests_done += 1;
            let elapsed_seconds = measurement_started_at.elapsed().as_secs_f64();
            if elapsed_seconds - last_report_seconds >= PROGRESS_INTERVAL_SECONDS {
                last_report_seconds = elapsed_seconds;
                eprintln!(
                    "[clef-latency] progress requests={requests_done}/{request_total} percent={:.0} eta_seconds={:.0}",
                    100.0 * requests_done as f64 / request_total as f64,
                    elapsed_seconds / requests_done as f64 * (request_total - requests_done) as f64
                );
            }
        }
        request_milliseconds.sort_by(f64::total_cmp);
        let median_milliseconds = request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT / 2];
        eprintln!(
            "[clef-latency] questions={question_count} sequence_tokens={CLEF_MAXIMUM_SEQUENCE_TOKENS} median_milliseconds={median_milliseconds:.1} fastest_milliseconds={:.1} slowest_milliseconds={:.1} tokens_per_second={:.0} {}",
            request_milliseconds[0],
            request_milliseconds[TIMED_REQUESTS_PER_QUESTION_COUNT - 1],
            f64::from(CLEF_MAXIMUM_SEQUENCE_TOKENS) / (median_milliseconds / 1_000.0),
            peak_memory_report(peak_memory_bytes_after_request(&mut decision_engine)),
        );
    }
}
