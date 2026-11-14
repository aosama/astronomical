//! Prefill memory recovery must serve the request, not fail it (issue #1729).
//!
//! On the largest sparse model at a 37 GiB ceiling, the default 4,096-token
//! prefill chunk exceeds the MLX active-memory limit while prefill seats
//! complete expert layers. Recovery sheds every retained expert and freezes
//! retention at zero, so the retry streams experts and fits. The freeze used
//! to be undone by the retry's first residency-plan refresh: the retry
//! re-seated every complete layer, failed again, and the user saw "configured
//! prefill chunk of 4096 tokens exceeded the MLX active-memory limit". This
//! journey sends the same 7,053-token prompt twice with the default chunking
//! (the first reply's decode seats the residency the second prefill meets)
//! and asserts both requests are served.
//!
//! Serving the request is not the whole promise. The freeze used to hold
//! until decode, so a long prompt that hit a recovery read all of its
//! remaining chunks with nothing kept in memory: on this model at a 32 GiB
//! ceiling the engine sat at about 3.7 GB while every layer was re-read from
//! storage for every chunk, and the person saw a slow first request that
//! used a tenth of the memory they had granted. The second journey sends a
//! short request and then a long one, as a coding agent does, and asserts
//! that memory in use is back above half the ceiling while the long prompt
//! is still being read. It runs with everyday settings (prompt cache on,
//! performance attribution off), because that is where people meet this and
//! because attribution changes what the engine learns from a streamed chunk.

use std::{fs, path::Path};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use futures_util::StreamExt;
use serde_json::json;
use serial_test::serial;
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::support::{
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
const PROMPT_TOKENS: usize = 7_053;
const MAXIMUM_OUTPUT_TOKENS: u64 = 16;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");
// The ceiling the refill was traced at. It defines this acceptance cell only;
// production code must not assume it.
const REFILL_MEMORY_CEILING_BYTES: u64 = 32 * (1 << 30);
const REFILL_WARM_UP_PROMPT_TOKENS: usize = 450;
// Four default chunks, so at least two run after the first full one.
const REFILL_LONG_PROMPT_TOKENS: usize = 14_000;
// Past this many tokens the first full chunk has completed and been
// measured, whether or not it needed a recovery.
const REFILL_MEASURED_PROMPT_TOKENS: u64 = 8_192;
const REFILL_STATUS_POLL_INTERVAL: Duration = Duration::from_millis(100);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the largest sparse model at 37 GB and checks prefill memory recovery serves the request"]
#[serial]
async fn should_serve_a_prompt_whose_first_prefill_chunk_exceeds_memory() {
    timeout(
        JOURNEY_BUDGET,
        serve_prompt_through_prefill_memory_recovery(),
    )
    .await
    .expect("the prefill memory-recovery journey must finish inside the 120-second runner budget");
}

async fn serve_prompt_through_prefill_memory_recovery() {
    let model_id = crate::support::largest_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let evidence_directory =
        crate::support::acceptance_evidence::acceptance_evidence_root("prefill-memory-recovery");
    fs::create_dir_all(&evidence_directory).expect("the evidence root should exist");
    let worker_home = tempfile::Builder::new()
        .prefix("worker-")
        .tempdir_in(&evidence_directory)
        .expect("the isolated worker home should exist")
        .keep();
    write_default_chunking_configuration(&worker_home, &model_directory);
    let prompt = crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        PROMPT_TOKENS,
    );
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &worker_home,
        sparse_decode_full_fit::MEMORY_CEILING_BYTES,
    )
    .await;
    let client = LocalOpenAiClient::new(server.server_address, "prefill-memory-recovery");
    let request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": MAXIMUM_OUTPUT_TOKENS,
        "temperature": 0,
    });
    // The first reply's decode seats complete expert layers; the second
    // request's 4,096-token prefill chunk then meets that residency, which
    // is the shape that failed in the field.
    let mut served_request_count = 0_usize;
    for _request_round in 0..2 {
        let stream = client
            .create_streaming_chat_completion(&request)
            .await
            .expect("the request should start");
        if consume_completion(stream).await {
            served_request_count += 1;
        }
    }
    stop_real_model_rest_server(server).await;
    assert_eq!(
        served_request_count,
        2,
        "both requests must be served through prefill memory recovery, not rejected; evidence in {}",
        worker_home.display()
    );
}

/// True when the stream finished with a finish reason and carried no error.
async fn consume_completion(mut stream: ChatCompletionStream) -> bool {
    let mut finished = false;
    while let Some(fragment) = stream.next().await {
        let fragment = fragment.expect("the completion stream must stay readable");
        if !fragment["error"].is_null() {
            println!(
                "[prefill-memory-recovery] request failed: {}",
                fragment["error"]
            );
            return false;
        }
        for choice in fragment["choices"].as_array().into_iter().flatten() {
            finished |= matches!(choice["finish_reason"].as_str(), Some("stop" | "length"));
        }
    }
    finished
}

/// The 37 GiB cell's configuration with the chunk sizes left at their
/// defaults, so the journey exercises what users actually get.
fn write_default_chunking_configuration(worker_home: &Path, model_directory: &Path) {
    let mut configuration_document =
        sparse_decode_full_fit::configuration_document(model_directory);
    if let Some(configuration_object) = configuration_document.as_object_mut() {
        configuration_object.remove("chunking");
    }
    // The largest model may live in a Hugging Face hub cache; scanning from
    // the configured roots names it by repository, as serving does.
    configuration_document["runtime"]["model_directories"] =
        json!(crate::support::configured_model_directory_roots());
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the largest sparse model at 32 GB and checks a long first prompt gets its memory back after a prefill recovery"]
#[serial]
async fn should_keep_experts_in_memory_again_while_a_long_prompt_is_still_being_read() {
    timeout(JOURNEY_BUDGET, read_a_long_prompt_after_a_short_one())
        .await
        .expect(
            "the prefill memory-refill journey must finish inside the 120-second runner budget",
        );
}

async fn read_a_long_prompt_after_a_short_one() {
    let model_id = crate::support::largest_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let evidence_directory =
        crate::support::acceptance_evidence::acceptance_evidence_root("prefill-memory-refill");
    fs::create_dir_all(&evidence_directory).expect("the evidence root should exist");
    let worker_home = tempfile::Builder::new()
        .prefix("worker-")
        .tempdir_in(&evidence_directory)
        .expect("the isolated worker home should exist")
        .keep();
    write_everyday_configuration(&worker_home);
    let warm_up_prompt = story_prompt(&model_directory, REFILL_WARM_UP_PROMPT_TOKENS);
    let long_prompt = story_prompt(&model_directory, REFILL_LONG_PROMPT_TOKENS);
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &worker_home,
        REFILL_MEMORY_CEILING_BYTES,
    )
    .await;
    let server_address = server.server_address;
    let client = LocalOpenAiClient::new(server_address, "prefill-memory-refill");
    // A short request first seats complete expert layers, which is the
    // residency the long prompt's first full chunk then has no room beside.
    let warm_up_stream = client
        .create_streaming_chat_completion(&story_request(model_id, &warm_up_prompt))
        .await
        .expect("the short request should start");
    assert!(
        consume_completion(warm_up_stream).await,
        "the short request must be served"
    );
    let long_stream = client
        .create_streaming_chat_completion(&story_request(model_id, &long_prompt))
        .await
        .expect("the long request should start");
    let (long_request_was_served, prefill_memory_observations) = tokio::join!(
        consume_completion(long_stream),
        observe_prefill_memory(server_address),
    );
    stop_real_model_rest_server(server).await;
    assert!(
        long_request_was_served,
        "the long request must be served; evidence in {}",
        worker_home.display()
    );
    assert_memory_was_refilled_during_prefill(&prefill_memory_observations, &worker_home);
}

/// The settings a fresh install runs with: the prompt cache is on and
/// performance attribution is off. Only the model folders are named, and the
/// log level is raised so the evidence folder records when the freeze lifts.
fn write_everyday_configuration(worker_home: &Path) {
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {"model_directories": crate::support::configured_model_directory_roots()},
        "diagnostics": {"log_level": "info"},
    });
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}

fn story_prompt(model_directory: &Path, prompt_token_count: usize) -> String {
    crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        prompt_token_count,
    )
}

fn story_request(model_id: &str, prompt: &str) -> serde_json::Value {
    json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": MAXIMUM_OUTPUT_TOKENS,
        "temperature": 0,
    })
}

/// One reading of the engine's status while it was reading the prompt.
#[derive(Clone, Copy, Debug)]
struct PrefillMemoryObservation {
    processed_prompt_tokens: u64,
    active_memory_bytes: u64,
}

/// Polls the status endpoint until the long request is over and returns every
/// distinct reading taken while the prompt was being read.
async fn observe_prefill_memory(
    server_address: std::net::SocketAddr,
) -> Vec<PrefillMemoryObservation> {
    let deadline = Instant::now() + JOURNEY_BUDGET;
    let mut prefill_memory_observations: Vec<PrefillMemoryObservation> = Vec::new();
    let mut request_was_seen = false;
    while Instant::now() < deadline {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        request_was_seen |= activity != "idle";
        if request_was_seen && activity == "idle" {
            break;
        }
        let memory_snapshot = &status_document["mlx_memory_snapshot"];
        if activity == "prompt_processing" && memory_snapshot["source"].as_str() == Some("prefill")
        {
            let observation = PrefillMemoryObservation {
                processed_prompt_tokens: status_document["progress"]["processed_tokens"]
                    .as_u64()
                    .unwrap_or(0),
                active_memory_bytes: memory_snapshot["active_memory_bytes"].as_u64().unwrap_or(0),
            };
            record_distinct_observation(&mut prefill_memory_observations, observation);
        }
        sleep(REFILL_STATUS_POLL_INTERVAL).await;
    }
    prefill_memory_observations
}

fn record_distinct_observation(
    prefill_memory_observations: &mut Vec<PrefillMemoryObservation>,
    observation: PrefillMemoryObservation,
) {
    let repeats_the_last_reading = prefill_memory_observations.last().is_some_and(|last| {
        last.processed_prompt_tokens == observation.processed_prompt_tokens
            && last.active_memory_bytes == observation.active_memory_bytes
    });
    if repeats_the_last_reading {
        return;
    }
    println!(
        "[prefill-memory-refill] processed_prompt_tokens={} active_memory_gb={:.1}",
        observation.processed_prompt_tokens,
        observation.active_memory_bytes as f64 / 1e9
    );
    prefill_memory_observations.push(observation);
}

/// Once a full chunk has been read and measured, the rest of the prompt must
/// not run with the granted memory left empty.
fn assert_memory_was_refilled_during_prefill(
    prefill_memory_observations: &[PrefillMemoryObservation],
    worker_home: &Path,
) {
    let highest_active_memory_bytes_after_measurement = prefill_memory_observations
        .iter()
        .filter(|observation| observation.processed_prompt_tokens >= REFILL_MEASURED_PROMPT_TOKENS)
        .map(|observation| observation.active_memory_bytes)
        .max();
    let Some(highest_active_memory_bytes_after_measurement) =
        highest_active_memory_bytes_after_measurement
    else {
        panic!(
            "no status reading was taken past {REFILL_MEASURED_PROMPT_TOKENS} prompt tokens; readings: {prefill_memory_observations:?}"
        );
    };
    assert!(
        highest_active_memory_bytes_after_measurement >= REFILL_MEMORY_CEILING_BYTES / 2,
        "after the first full chunk the engine held at most {:.1} GB of a {:.1} GB ceiling while still reading the prompt; readings: {prefill_memory_observations:?}; evidence in {}",
        highest_active_memory_bytes_after_measurement as f64 / 1e9,
        REFILL_MEMORY_CEILING_BYTES as f64 / 1e9,
        worker_home.display()
    );
}
