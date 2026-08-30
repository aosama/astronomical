//! User journey proving that a fitting model stays completely resident.
//!
//! At a 35 GB decimal-SI MLX ceiling, the resident sparse MoE e2e fixture has enough room
//! for model core, every expert, and required request headroom. Cache is enabled.
//! The journey sends a 15,000-token Romeo and Juliet request and proves:
//!
//! - generation completes through the production REST/server/worker stack;
//! - every active phase stays fully in RAM;
//! - the generation attribution report contains zero expert positional-read bytes.
//!
use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::support::serving_rest::{
    get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

fn model_id() -> &'static str {
    crate::support::resident_sparse_moe_model_id()
}
// Acceptance cells are explicit evidence, not production constants. Runtime
// policy continues to derive capacity from user/machine ceiling and model geometry.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 35_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 15_000;
// Must cover thinking + the 24-token think-end transition + one visible token.
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 512;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 256;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const REQUEST_MUST_BECOME_ACTIVE_WITHIN: Duration = Duration::from_secs(20);
const JOURNEY_DEADLINE: Duration = Duration::from_secs(120);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept complete expert residency without request-time SSD streaming"]
async fn should_keep_all_experts_resident_and_avoid_ssd_reads_when_the_model_fits_memory() {
    timeout(
        JOURNEY_DEADLINE,
        run_complete_expert_residency_rest_journey(),
    )
    .await
    .expect("the complete expert-residency REST journey must finish within 120 seconds");
}

async fn run_complete_expert_residency_rest_journey() {
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home = isolated_complete_residency_worker_home();
    write_acceptance_config(&isolated_worker_home, &model_directory);
    // One 15,000-token slice of the long Romeo and Juliet fixture. The builder
    // tokenizes with the real model tokenizer so the public request is exact.
    let user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        &isolated_worker_home,
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-acceptance-client"),
    );
    let completion_request = json!({
        "model": model_id(),
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "temperature": 1,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    eprintln!(
        "[complete-expert-residency] status=progress phase=request_send prompt_characters={}",
        user_message.len()
    );
    let streamed_completion: StreamResponse<Value> = timeout(
        Duration::from_secs(60),
        openai_client.chat().create_stream_byot(completion_request),
    )
    .await
    .expect("the 15,000-token REST request must be accepted within 60 seconds")
    .expect("the complete-residency REST request should start");
    eprintln!("[complete-expert-residency] status=progress phase=stream_open");
    let (completed_stream, final_status) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_resident_request_until_idle(server_address, &logging_directory),
    );
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    let average_prefill_tokens_per_second =
        final_status["serving_session"]["average_prefill_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average prefill throughput");
    let average_generation_tokens_per_second =
        final_status["serving_session"]["average_generation_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average generation throughput");
    stop_real_model_rest_server(real_model_rest_server).await;
    assert_eq!(
        final_status["expert_memory_mode"].as_str(),
        Some("resident"),
        "the completed request must leave all experts resident: {final_status}"
    );
    let expert_source_read_bytes = generation_expert_source_read_bytes(&isolated_worker_home);
    assert_eq!(
        expert_source_read_bytes, 0,
        "a completely resident model must not stream expert ranges from SSD during the request"
    );
    eprintln!(
        "[complete-expert-residency] status=success prompt_tokens={PROMPT_TOKEN_COUNT} expert_memory_mode=resident expert_source_read_bytes={expert_source_read_bytes} average_prefill_tokens_per_second={average_prefill_tokens_per_second:.2} average_generation_tokens_per_second={average_generation_tokens_per_second:.2} output_characters={}",
        completed_stream.model_text.len(),
    );
}

async fn observe_resident_request_until_idle(
    server_address: SocketAddr,
    logging_directory: &Path,
) -> Value {
    let request_started_at = Instant::now();
    let deadline = request_started_at + JOURNEY_DEADLINE;
    let mut observed_active_request = false;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            let activity = status_document["activity"].as_str().unwrap_or("unknown");
            let expert_memory_mode = status_document["expert_memory_mode"]
                .as_str()
                .unwrap_or("unavailable");
            let processed_tokens = status_document["progress"]["processed_tokens"]
                .as_u64()
                .unwrap_or(0);
            let total_tokens = status_document["progress"]["total_tokens"]
                .as_u64()
                .unwrap_or(0);
            let expert_payload_bytes =
                status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
                    .as_u64()
                    .unwrap_or(0);
            eprintln!(
                "[complete-expert-residency] status=progress activity={activity} expert_memory_mode={expert_memory_mode} processed_tokens={processed_tokens} total_tokens={total_tokens} expert_payload_bytes={expert_payload_bytes}"
            );
            last_status_log_at = Instant::now();
        }
        if status_document["activity"] != "idle" {
            observed_active_request = true;
            assert_eq!(
                status_document["expert_memory_mode"].as_str(),
                Some("resident"),
                "a fitting model must stay fully in RAM while the request is active: {status_document}"
            );
        }
        if !observed_active_request
            && request_started_at.elapsed() >= REQUEST_MUST_BECOME_ACTIVE_WITHIN
        {
            print_worker_diagnostic_logs(logging_directory);
            panic!(
                "the 15,000-token request stayed idle for {} seconds; prompt processing never started: {status_document}",
                REQUEST_MUST_BECOME_ACTIVE_WITHIN.as_secs()
            );
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_active_request
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return status_document;
        }
        assert!(
            Instant::now() < deadline,
            "the complete-residency REST journey did not return to idle: {status_document}"
        );
        sleep(Duration::from_millis(100)).await;
    }
}

struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
}

async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> CompletedStream {
    let mut streamed_model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = match stream_item {
            Ok(stream_chunk) => stream_chunk,
            Err(stream_error) => {
                eprintln!("[complete-expert-residency] status=stream_error error={stream_error}");
                break;
            }
        };
        if !stream_chunk["error"].is_null() {
            panic!("the 15,000-token REST stream returned an error: {stream_chunk}");
        }
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                streamed_model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    CompletedStream {
        model_text: streamed_model_text.trim().to_owned(),
        finish_reason,
    }
}

fn generation_expert_source_read_bytes(isolated_worker_home: &Path) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the acceptance journey should write performance attribution");
    let generation_reports = attribution_log
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each performance-attribution row should be valid JSON")
        })
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .collect::<Vec<_>>();
    assert_eq!(
        generation_reports.len(),
        1,
        "the single REST request should produce one generation report"
    );
    // `positional_file_read_byte_count` is logical executed pread payload. Zero is
    // strong enough for this journey: with every expert resident there should be
    // no expert source operation for the operating system to cache or service.
    generation_reports[0]["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|counter_report| counter_report["counter"] == "positional_file_read_byte_count")
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the complete-residency configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": MAXIMUM_MLX_MEMORY_BYTES / 1_000_000_000,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "prompt_cache_max_size_gb": 50,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the complete-residency configuration should serialize"),
    )
    .expect("the complete-residency configuration should be written");
}

fn isolated_complete_residency_worker_home() -> PathBuf {
    // Persistent across a panic so worker logs remain after a fail-fast abort.
    // The path is process-temp, not a developer home directory.
    let worker_home = std::env::temp_dir().join("astronomical-complete-expert-residency-e2e");
    let _ = fs::remove_dir_all(&worker_home);
    fs::create_dir_all(&worker_home).expect("the complete-residency worker home should be created");
    worker_home
        .canonicalize()
        .expect("the complete-residency worker home should canonicalize")
}

fn print_worker_diagnostic_logs(logging_directory: &Path) {
    let Ok(directory_entries) = fs::read_dir(logging_directory) else {
        eprintln!(
            "[complete-expert-residency] status=worker_logs reason=unreadable path={}",
            logging_directory.display()
        );
        return;
    };
    let mut worker_log_paths = directory_entries
        .flatten()
        .map(|directory_entry| directory_entry.path())
        .filter(|log_path| {
            log_path
                .file_name()
                .and_then(|file_name| file_name.to_str())
                .is_some_and(|file_name| file_name.ends_with(".log"))
        })
        .collect::<Vec<_>>();
    worker_log_paths.sort();
    if worker_log_paths.is_empty() {
        eprintln!("[complete-expert-residency] status=worker_logs reason=missing");
        return;
    }
    for worker_log_path in worker_log_paths {
        let Ok(log_contents) = fs::read_to_string(&worker_log_path) else {
            continue;
        };
        eprintln!(
            "[complete-expert-residency] status=worker_log path={}",
            worker_log_path.display()
        );
        for log_line in log_contents.lines() {
            if log_line.contains("ERROR")
                || log_line.contains("error")
                || log_line.contains("FATAL")
                || log_line.contains("fatal")
                || log_line.contains("panic")
                || log_line.contains("Generate")
                || log_line.contains("generation")
                || log_line.contains("admission")
                || log_line.contains("prompt-cache")
                || log_line.contains("demot")
                || log_line.contains("resident")
            {
                eprintln!("[complete-expert-residency] status=worker_trace line={log_line}");
            }
        }
    }
}
