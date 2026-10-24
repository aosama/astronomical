//! The screenshot regression cell: paged multi-chunk prefill, then full-fit decode.

use std::{fs, net::SocketAddr, path::Path, time::Instant};

use astronomical_test_harness::{sparse_decode_full_fit, test_run::TestRunContext};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, sleep, timeout};

use crate::support::{
    openai_client::{ChatCompletionStream, LocalOpenAiClient},
    serving_rest::{get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server},
};

const JOURNEY_BUDGET: Duration = Duration::from_secs(115);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the large sparse model at 37 GB and checks paged prefill becomes complete decode residency"]
#[serial]
async fn should_complete_partial_prefill_residency_before_decode_at_37_gigabytes() {
    timeout(JOURNEY_BUDGET, run_decode_full_fit_journey())
        .await
        .expect("the full-fit decode journey must finish inside the 120-second runner budget");
}

async fn run_decode_full_fit_journey() {
    let model_id = crate::support::large_sparse_moe_streaming_model_id();
    let model_directory = TestRunContext::current()
        .configuration
        .model_directory
        .unwrap_or_else(|| crate::support::configured_installed_model_directory_by_id(model_id));
    let evidence_directory =
        crate::support::acceptance_evidence::acceptance_evidence_root("decode-full-fit");
    fs::create_dir_all(&evidence_directory).expect("the evidence root should exist");
    let worker_home = tempfile::Builder::new()
        .prefix("worker-")
        .tempdir_in(&evidence_directory)
        .expect("the isolated worker home should exist")
        .keep();
    write_configuration(&worker_home, &model_directory);
    let prompt = crate::support::exact_model_prompt::build_model_prompt_content_at_most(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Continue the supplied Romeo and Juliet story.",
        26_552,
    );
    let server = launch_real_model_rest_server(
        model_id,
        model_directory,
        &worker_home,
        sparse_decode_full_fit::MEMORY_CEILING_BYTES,
    )
    .await;
    let address = server.server_address;
    let initial_status = get_json_endpoint(address, "/v1/status").await;
    assert_eq!(
        initial_status["mlx_memory_ceiling_bytes"],
        sparse_decode_full_fit::MEMORY_CEILING_BYTES
    );
    let client = LocalOpenAiClient::new(address, "decode-full-fit-acceptance");
    let request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": 128,
        "temperature": 1,
        "thinking_budget": 32
    });
    let stream = client
        .create_streaming_chat_completion(&request)
        .await
        .expect("the full-fit request should start");
    let (output_text, decode_snapshots) = tokio::join!(
        consume_completion(stream),
        observe_decode_snapshots(address),
    );
    stop_real_model_rest_server(server).await;
    fs::write(
        worker_home.join("decode-status-samples.json"),
        serde_json::to_vec_pretty(&decode_snapshots).expect("status evidence should serialize"),
    )
    .expect("status evidence should be retained");
    assert!(!output_text.is_empty(), "the real model must produce text");
    assert!(
        !decode_snapshots.is_empty(),
        "the journey must observe decode, not just idle residency"
    );
    for snapshot in &decode_snapshots {
        sparse_decode_full_fit::validate_decode_residency(snapshot)
            .unwrap_or_else(|failure| panic!("{failure}; evidence={}", worker_home.display()));
    }
    let attribution_log = fs::read_to_string(
        worker_home.join(".astronomical-dev/logs/performance-attribution.jsonl"),
    )
    .expect("the worker should flush attribution before shutdown");
    let reports = attribution_log
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).expect("each attribution record should parse")
        })
        .filter(|report| report["report_kind"] == "generation")
        // Resident admission may emit an empty report before demoting to paging.
        .filter(|report| {
            sparse_decode_full_fit::attribution_counter(report, "generated_token_count")
                .expect("the generated-token counter should be valid")
                > 0
        })
        .collect::<Vec<_>>();
    assert_eq!(reports.len(), 1, "this journey serves exactly one request");
    sparse_decode_full_fit::validate_generation_attribution(&reports[0])
        .unwrap_or_else(|failure| panic!("{failure}; evidence={}", worker_home.display()));
}

async fn consume_completion(mut stream: ChatCompletionStream) -> String {
    let mut output_text = String::new();
    let mut finished = false;
    while let Some(fragment) = stream.next().await {
        let fragment = fragment.expect("the completion stream must not fail");
        assert!(fragment["error"].is_null(), "generation failed: {fragment}");
        for choice in fragment["choices"].as_array().into_iter().flatten() {
            for text_field in ["content", "reasoning_content"] {
                if let Some(text) = choice["delta"][text_field].as_str() {
                    output_text.push_str(text);
                }
            }
            finished |= matches!(choice["finish_reason"].as_str(), Some("stop" | "length"));
        }
    }
    assert!(finished, "the completion stream must carry a finish reason");
    output_text
}

async fn observe_decode_snapshots(address: SocketAddr) -> Vec<Value> {
    let started_at = Instant::now();
    let mut last_progress_at = started_at;
    let mut saw_request = false;
    let mut decode_snapshots = Vec::new();
    loop {
        let status = get_json_endpoint(address, "/v1/status").await;
        let activity = status["activity"]
            .as_str()
            .expect("status must contain activity");
        saw_request |= activity != "idle";
        if activity == "generating" && status["mlx_memory_snapshot"]["source"] == "decode_submitted"
        {
            decode_snapshots.push(status.clone());
        }
        if last_progress_at.elapsed() >= Duration::from_secs(5) {
            let elapsed_seconds = started_at.elapsed().as_secs();
            let remaining_seconds = JOURNEY_BUDGET.as_secs().saturating_sub(elapsed_seconds);
            eprintln!(
                "[decode-full-fit] activity={activity} progress_percent={} eta_seconds={remaining_seconds} expert_payload_bytes={}",
                elapsed_seconds * 100 / JOURNEY_BUDGET.as_secs(),
                status["mlx_memory_snapshot"]["expert_payload_bytes"],
            );
            last_progress_at = Instant::now();
        }
        if saw_request && activity == "idle" {
            break;
        }
        assert!(
            started_at.elapsed() < JOURNEY_BUDGET,
            "decode never completed: {status}"
        );
        sleep(Duration::from_millis(50)).await;
    }
    decode_snapshots
}

fn write_configuration(worker_home: &Path, model_directory: &Path) {
    let state_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&state_directory).expect("the state directory should exist");
    fs::write(
        state_directory.join("config.json"),
        serde_json::to_vec_pretty(&sparse_decode_full_fit::configuration_document(
            model_directory,
        ))
        .expect("the configuration should serialize"),
    )
    .expect("the configuration should be written");
}
