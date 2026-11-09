//! **Performance measurement — run on demand, not in regression runs.**
//! Run with `cargo measure-paging-submission-intervals --configuration PATH`
//! when choosing the `experimental_ssd_paging_*_graph_submission_layer_interval`
//! defaults; the regression lane (`cargo test-acceptance`) does not build this
//! target's measurements.
//!
//! Graph-submission layer intervals while experts stream from SSD.
//!
//! The interval is how many decoder layers accumulate before the forward is
//! handed to the graphics processor. One cell measures one pair of intervals,
//! taken from the `paging_experiment` object of the configuration file, so a
//! sweep is several runs that differ only in that object:
//!
//! ```json
//! {"paging_experiment": {"maximum_mlx_memory_gigabytes": 14,
//!   "streaming_prompt_processing_chunk_tokens": 4096,
//!   "prefill_graph_submission_layer_interval": 1,
//!   "generation_graph_submission_layer_interval": 3}}
//! ```
//!
//! An interval above the decoder layer count never submits inside a forward,
//! which is how a cell measures "no intermediate submission" (the harness
//! rejects zero).
//!
//! The cell uses the smaller 4-bit sparse model under a lowered ceiling: it
//! forces layers to stream while leaving most of the machine's RAM to the
//! operating system, so timings are not distorted by swapping. Pick the
//! ceiling by what the cell must exercise. Most of this model's experts are
//! resident at 20 GiB (18.1 of 19.6 GB), so decode reads little from SSD
//! there and far more at 14 GiB; the report prints the decode source bytes so
//! a cell proves which case it ran.
//!
//! Performance attribution stays off: it synchronizes the graphics processor
//! after every stage, which would add the very hand-offs being compared.
//!
//! Measured with a 10,240-token prompt, 500 output tokens and 4,096-token
//! chunks; run-to-run spread was about two percent:
//!
//! - Generation interval at 14 GiB (decode read about 114 GB from SSD): 1, 2,
//!   3, 4 and 6 decoded at 33.5 to 33.9 tok/s, so the value is not sensitive.
//! - Generation interval at 20 GiB (decode read about 26 GB): 1, 3 and 6
//!   decoded at 60.0 to 60.6 tok/s, and no intermediate submission at 56.8,
//!   twice. Submitting inside the forward is worth six percent there.
//! - Prefill interval at 14 GiB: 1 averaged 2,190 tok/s over six runs and 2
//!   averaged 2,050 over two. At 20 GiB both were about 2,300. Peak memory was
//!   the same for every interval at each ceiling.
//!
//! The defaults (prefill 1, generation 3) therefore stay as they are.

use std::{fs, path::Path};

use astronomical_test_harness::{
    sparse_decode_full_fit,
    test_run::{PagingExperimentCase, TestRunContext},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, Instant, timeout};

use crate::support::{
    exact_model_prompt::build_model_prompt_content_at_most,
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server},
};

/// Five seconds under the runner's 120-second budget, so an overrun fails
/// with this cell's own message instead of the runner ending the process.
const CELL_BUDGET: Duration = Duration::from_secs(115);
const WARM_UP_PROMPT_TOKENS: usize = 500;
const WARM_UP_OUTPUT_TOKENS: u64 = 50;
const MEASURED_PROMPT_TOKENS: usize = 10_240;
const MEASURED_OUTPUT_TOKENS: u64 = 500;
/// The repository's performance rule allows ten percent either way.
const MINIMUM_MEASURED_OUTPUT_TOKENS: u64 = 450;
const STATS_POLL_ATTEMPTS: usize = 50;
const STATS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const BYTES_PER_GIGABYTE: f64 = 1e9;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");
/// The two instructions differ from their first word so the measured request
/// cannot adopt the warm-up's decoder state as a shared prefix.
const WARM_UP_INSTRUCTION: &str = "Name the two feuding families in the supplied story.";
const MEASURED_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story.";

struct TimedCompletion {
    prompt_token_count: u64,
    completion_token_count: u64,
    first_token_elapsed: Duration,
    last_token_elapsed: Duration,
}

impl TimedCompletion {
    /// Prompt tokens over the time to the first streamed token, so the rate
    /// includes expert streaming and every submission inside prefill.
    fn prefill_tokens_per_second(&self) -> f64 {
        self.prompt_token_count as f64 / self.first_token_elapsed.as_secs_f64()
    }

    /// The first token belongs to prefill, so decode covers the rest.
    fn decode_tokens_per_second(&self) -> f64 {
        let decode_elapsed = self
            .last_token_elapsed
            .saturating_sub(self.first_token_elapsed);
        if decode_elapsed.is_zero() {
            // A one-token reply has no decode interval; zero keeps the report
            // line finite instead of printing the result of zero over zero.
            return 0.0;
        }
        self.completion_token_count.saturating_sub(1) as f64 / decode_elapsed.as_secs_f64()
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the 4-bit sparse model under the configured ceiling and measures serving rates for the configured SSD-paging graph-submission intervals"]
#[serial]
async fn should_measure_serving_rates_for_the_configured_graph_submission_intervals() {
    timeout(CELL_BUDGET, measure_configured_interval_cell())
        .await
        .expect("the interval cell must finish inside the 120-second runner budget");
}

async fn measure_configured_interval_cell() {
    let test_run_configuration = TestRunContext::current().configuration;
    let paging_experiment = test_run_configuration.paging_experiment;
    let ceiling_bytes = paging_experiment
        .maximum_mlx_memory_bytes()
        .expect("the interval cell needs a valid binary-gigabyte ceiling");
    // A configured model directory (for example an Expert Atlas build of the
    // same model) replaces the default 4-bit sparse model; its folder name is
    // then the model id. The default keeps its registered id, because a
    // discovered directory can be a snapshot folder whose name is a hash.
    let (model_directory, model_id) = match test_run_configuration.model_directory {
        Some(configured_model_directory) => {
            let configured_model_id = configured_model_directory
                .file_name()
                .and_then(|directory_name| directory_name.to_str())
                .expect("the model directory has a UTF-8 name")
                .to_owned();
            (configured_model_directory, configured_model_id)
        }
        None => {
            let default_model_id = crate::support::resident_sparse_moe_model_id();
            (
                crate::support::configured_installed_model_directory_by_id(default_model_id),
                default_model_id.to_owned(),
            )
        }
    };
    let model_id = model_id.as_str();
    let worker_home = tempfile::tempdir().expect("the isolated worker home should exist");
    write_configuration(worker_home.path(), &model_directory, &paging_experiment);
    let warm_up_prompt = build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        WARM_UP_INSTRUCTION,
        WARM_UP_PROMPT_TOKENS,
    );
    let measured_prompt = build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        MEASURED_INSTRUCTION,
        MEASURED_PROMPT_TOKENS,
    );
    let server =
        launch_real_model_rest_server(model_id, model_directory, worker_home.path(), ceiling_bytes)
            .await;
    let client = LocalOpenAiClient::new(server.server_address, "graph-submission-interval");
    // The warm-up pays for Metal pipeline compilation and the first expert
    // seating, which would otherwise be charged to the measured prefill.
    send_timed_request(&client, model_id, &warm_up_prompt, WARM_UP_OUTPUT_TOKENS).await;
    let measured_completion =
        send_timed_request(&client, model_id, &measured_prompt, MEASURED_OUTPUT_TOKENS).await;
    let expert_stats = poll_until_stats_available(server.server_address).await;
    let server_status = get_json_endpoint(server.server_address, "/v1/status").await;
    stop_real_model_rest_server(server).await;
    report_and_check(
        &paging_experiment,
        &measured_completion,
        &expert_stats,
        &server_status,
        ceiling_bytes,
    );
}

async fn send_timed_request(
    client: &LocalOpenAiClient,
    model_id: &str,
    prompt: &str,
    maximum_output_tokens: u64,
) -> TimedCompletion {
    let completion_request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": maximum_output_tokens,
        "temperature": 0,
    });
    let request_started_at = Instant::now();
    let completion_stream = client
        .create_streaming_chat_completion(&completion_request)
        .await
        .expect("the request should start");
    consume_completion_with_timing(completion_stream, request_started_at).await
}

async fn consume_completion_with_timing(
    mut completion_stream: ChatCompletionStream,
    request_started_at: Instant,
) -> TimedCompletion {
    let mut usage = Value::Null;
    let mut first_token_elapsed = None;
    let mut last_token_elapsed = Duration::ZERO;
    while let Some(stream_fragment) = completion_stream.next().await {
        let stream_fragment = stream_fragment.expect("the completion stream must not fail");
        assert!(
            stream_fragment["error"].is_null(),
            "generation failed: {stream_fragment}"
        );
        if stream_fragment["usage"].is_object() {
            usage = stream_fragment["usage"].clone();
        }
        if fragment_carries_generated_text(&stream_fragment) {
            last_token_elapsed = request_started_at.elapsed();
            if first_token_elapsed.is_none() {
                first_token_elapsed = Some(last_token_elapsed);
            }
        }
    }
    TimedCompletion {
        prompt_token_count: usage["prompt_tokens"]
            .as_u64()
            .expect("the final usage should report prompt tokens"),
        completion_token_count: usage["completion_tokens"]
            .as_u64()
            .expect("the final usage should report completion tokens"),
        first_token_elapsed: first_token_elapsed
            .expect("the completion should stream at least one token"),
        last_token_elapsed,
    }
}

/// Reasoning text counts: it is decoded by the same forward as answer text.
fn fragment_carries_generated_text(stream_fragment: &Value) -> bool {
    for choice in stream_fragment["choices"].as_array().into_iter().flatten() {
        let answer_text = choice["delta"]["content"].as_str().unwrap_or_default();
        let reasoning_text = choice["delta"]["reasoning_content"]
            .as_str()
            .unwrap_or_default();
        if !answer_text.is_empty() || !reasoning_text.is_empty() {
            return true;
        }
    }
    false
}

async fn poll_until_stats_available(server_address: std::net::SocketAddr) -> Value {
    for _poll_attempt in 0..STATS_POLL_ATTEMPTS {
        let expert_stats = get_json_endpoint(server_address, "/v1/experts/stats").await;
        if expert_stats["available"] == json!(true) {
            return expert_stats;
        }
        tokio::time::sleep(STATS_POLL_INTERVAL).await;
    }
    panic!("the expert stats must become available after the measured reply finishes");
}

fn report_and_check(
    paging_experiment: &PagingExperimentCase,
    measured_completion: &TimedCompletion,
    expert_stats: &Value,
    server_status: &Value,
    ceiling_bytes: u64,
) {
    let peak_memory_bytes = unsigned_field(server_status, "/mlx_memory_snapshot/peak_memory_bytes");
    // The whole document, so two cells whose rates differ can be compared
    // field by field without rerunning them.
    println!("[graph-submission-interval] expert_stats={expert_stats}");
    println!(
        "[graph-submission-interval] ceiling_gib={} chunk_tokens={} prefill_interval={} generation_interval={} prompt_tokens={} completion_tokens={} prefill_tokens_per_second={:.1} decode_tokens_per_second={:.2} peak_gb={:.2} expert_memory_mode={} resident_expert_payload_gb={:.2} complete_expert_payload_gb={:.2} prefill_source_gb={:.2} decode_source_gb={:.2}",
        paging_experiment.maximum_mlx_memory_gigabytes,
        paging_experiment.streaming_prompt_processing_chunk_tokens,
        paging_experiment.prefill_graph_submission_layer_interval,
        paging_experiment.generation_graph_submission_layer_interval,
        measured_completion.prompt_token_count,
        measured_completion.completion_token_count,
        measured_completion.prefill_tokens_per_second(),
        measured_completion.decode_tokens_per_second(),
        peak_memory_bytes as f64 / BYTES_PER_GIGABYTE,
        expert_stats["expert_memory_mode"],
        gigabytes_field(expert_stats, "/residency/resident_expert_payload_bytes"),
        gigabytes_field(
            expert_stats,
            "/residency/complete_expert_payload_byte_count"
        ),
        gigabytes_field(expert_stats, "/value/mandatory_prefill_source_bytes"),
        gigabytes_field(expert_stats, "/value/mandatory_decode_source_bytes"),
    );
    assert!(
        measured_completion.completion_token_count >= MINIMUM_MEASURED_OUTPUT_TOKENS,
        "decode must cover about {MEASURED_OUTPUT_TOKENS} tokens to be a fair rate"
    );
    assert!(
        gigabytes_field(expert_stats, "/value/mandatory_prefill_source_bytes") > 0.0,
        "the cell must stream experts from SSD during prefill, or the intervals are never used"
    );
    assert!(
        peak_memory_bytes <= ceiling_bytes,
        "the cell exceeded the MLX memory ceiling"
    );
}

/// A missing field fails the cell: reading it as zero would let the ceiling
/// check pass and print a rate line that looks complete.
fn unsigned_field(stats_document: &Value, field_path: &str) -> u64 {
    match stats_document.pointer(field_path).and_then(Value::as_u64) {
        Some(field_value) => field_value,
        None => panic!("the server reply must carry an unsigned {field_path}"),
    }
}

fn gigabytes_field(stats_document: &Value, field_path: &str) -> f64 {
    unsigned_field(stats_document, field_path) as f64 / BYTES_PER_GIGABYTE
}

fn write_configuration(
    worker_home: &Path,
    model_directory: &Path,
    paging_experiment: &PagingExperimentCase,
) {
    let mut configuration_document =
        sparse_decode_full_fit::configuration_document(model_directory);
    configuration_document["runtime"]["maximum_mlx_memory_gb"] =
        json!(paging_experiment.maximum_mlx_memory_gigabytes);
    configuration_document["diagnostics"]["performance_attribution_enabled"] = json!(false);
    let chunking = &mut configuration_document["chunking"];
    chunking["fixed_ssd_streaming_prompt_processing_chunk_size_tokens"] =
        json!(paging_experiment.streaming_prompt_processing_chunk_tokens);
    chunking["experimental_ssd_paging_prefill_graph_submission_layer_interval"] =
        json!(paging_experiment.prefill_graph_submission_layer_interval);
    chunking["experimental_ssd_paging_generation_graph_submission_layer_interval"] =
        json!(paging_experiment.generation_graph_submission_layer_interval);
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}

#[test]
fn should_count_a_fragment_with_answer_text_as_generated() {
    let answer_fragment = json!({"choices": [{"delta": {"content": "Two households"}}]});

    assert!(fragment_carries_generated_text(&answer_fragment));
}

#[test]
fn should_count_a_fragment_with_only_reasoning_text_as_generated() {
    let reasoning_fragment = json!({"choices": [{"delta": {"reasoning_content": "Verona"}}]});

    assert!(fragment_carries_generated_text(&reasoning_fragment));
}

#[test]
fn should_not_count_usage_or_finish_fragments_as_generated() {
    let finish_fragment = json!({"choices": [{"delta": {}, "finish_reason": "length"}]});
    let usage_fragment = json!({"choices": [], "usage": {"completion_tokens": 500}});

    assert!(!fragment_carries_generated_text(&finish_fragment));
    assert!(!fragment_carries_generated_text(&usage_fragment));
}

#[test]
fn should_leave_the_first_token_out_of_the_decode_rate() {
    let timed_completion = TimedCompletion {
        prompt_token_count: 10_000,
        completion_token_count: 501,
        first_token_elapsed: Duration::from_secs(5),
        last_token_elapsed: Duration::from_secs(15),
    };

    assert_eq!(timed_completion.prefill_tokens_per_second(), 2_000.0);
    assert_eq!(timed_completion.decode_tokens_per_second(), 50.0);
}

#[test]
fn should_report_a_zero_decode_rate_for_a_single_token_reply() {
    let single_token_completion = TimedCompletion {
        prompt_token_count: 10_000,
        completion_token_count: 1,
        first_token_elapsed: Duration::from_secs(5),
        last_token_elapsed: Duration::from_secs(5),
    };

    assert_eq!(single_token_completion.decode_tokens_per_second(), 0.0);
}

#[test]
fn should_read_a_present_stats_field_in_gigabytes() {
    let expert_stats = json!({"value": {"mandatory_decode_source_bytes": 26_500_000_000_u64}});

    assert_eq!(
        gigabytes_field(&expert_stats, "/value/mandatory_decode_source_bytes"),
        26.5
    );
}

#[test]
#[should_panic(expected = "must carry an unsigned /mlx_memory_snapshot/peak_memory_bytes")]
fn should_fail_when_a_stats_field_is_missing() {
    let status_without_snapshot = json!({"state": "ready"});

    unsigned_field(
        &status_without_snapshot,
        "/mlx_memory_snapshot/peak_memory_bytes",
    );
}
