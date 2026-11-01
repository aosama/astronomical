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

use std::{fs, path::Path};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use futures_util::StreamExt;
use serde_json::json;
use serial_test::serial;
use tokio::time::{Duration, timeout};

use crate::support::{
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
const PROMPT_TOKENS: usize = 7_053;
const MAXIMUM_OUTPUT_TOKENS: u64 = 16;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

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
