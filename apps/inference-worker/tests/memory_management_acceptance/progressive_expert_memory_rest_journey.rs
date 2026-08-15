//! User journey proving useful demand retention under a constrained 23 GB ceiling.
//!
//! The model cannot keep every expert resident in this qualification cell. The
//! desired behavior is therefore not merely "request succeeds": routed experts
//! must remain reusable across decoder layers instead of consuming nearly the
//! whole ceiling on early layers while repeatedly reading omitted routes.
//!
//! The journey observes public status while consuming a real streaming response,
//! then prints detailed admission evidence from isolated logs. The durable user
//! behavior is bounded memory plus decode demand retention across every layer.

use std::{fs, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
// This ceiling defines a reproducible acceptance cell only. Production code must
// not hardwire it or assume this model always leaves exactly four layers cold.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 23_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 7_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 1_280;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 256;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept expert-memory management behavior"]
async fn should_progressively_grow_expert_residency_during_prefill_and_restore_prefill_residency_after_generation()
 {
    timeout(
        JOURNEY_TIMEOUT,
        run_progressive_expert_memory_rest_journey(),
    )
    .await
    .expect("the progressive expert-memory REST journey must finish within 115 seconds");
}

async fn run_progressive_expert_memory_rest_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the memory-management worker home should be created");
    write_qualification_config(isolated_worker_home.path(), &model_directory);
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(3);
    let user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &repeated_source,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-qualification-client"),
    );
    let completion_request = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": user_message}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    let streamed_completion: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the public REST summary request should start");
    let (completed_stream, memory_evidence) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_progressive_expert_memory(server_address),
    );
    // Status polling runs concurrently with Server-Sent Events consumption so the
    // test observes the active request and its final bounded idle state.
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    let final_expert_payload_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
    let average_prefill_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_prefill_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average prefill throughput");
    let average_generation_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_generation_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average generation throughput");
    let final_active_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["active_memory_bytes"]
            .as_u64()
            .expect("the completed status should report final active MLX memory");
    let peak_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["peak_memory_bytes"]
            .as_u64()
            .expect("the completed status should report peak MLX memory");
    assert!(final_expert_payload_bytes > 0);
    assert!(final_active_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES);
    assert!(peak_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES.saturating_add(230_000_000));
    // Stop before reading logs to flush asynchronous tracing and attribution.
    stop_real_model_rest_server(real_model_rest_server).await;
    let memory_admission_decisions =
        memory_admission_decision_log_lines(isolated_worker_home.path());
    for memory_admission_decision in &memory_admission_decisions {
        eprintln!("[progressive-expert-memory] status=memory_decision {memory_admission_decision}");
    }
    let retained_expert_fill_decisions =
        retained_expert_fill_decision_log_lines(isolated_worker_home.path());
    for retained_expert_fill_decision in &retained_expert_fill_decisions {
        eprintln!(
            "[progressive-expert-memory] status=retained_fill_decision {retained_expert_fill_decision}"
        );
    }
    let expert_source_read_bytes = generation_expert_source_read_bytes(isolated_worker_home.path());
    let decode_streamed_layer_indices = decode_streamed_layer_indices(isolated_worker_home.path());
    // Prefill streams one complete expert model per chunk, then decode fill
    // rereads the retained subset. A healthy 7,000-plus-1,280-token cell with
    // zero decode-streamed layers measured 120.01 GB. Demand-per-byte ranking
    // measured 123.48 GB and must still fail.
    const MAXIMUM_WEIGHTED_RANKING_SOURCE_READ_BYTES: u64 = 122_000_000_000;
    assert!(
        expert_source_read_bytes < MAXIMUM_WEIGHTED_RANKING_SOURCE_READ_BYTES,
        "decode should reuse retained experts instead of continuously reading expert sources; observed {expert_source_read_bytes} logical source bytes across {} streamed decode layers",
        decode_streamed_layer_indices.len()
    );
    assert!(
        decode_streamed_layer_indices.len() < 40,
        "demand retention should prevent every decoder layer from streaming throughout decode; streamed {decode_streamed_layer_indices:?}"
    );
    assert_eq!(
        generation_attribution_report_count(isolated_worker_home.path()),
        1
    );
    eprintln!(
        "[progressive-expert-memory] status=success prompt_tokens={PROMPT_TOKEN_COUNT} maximum_mlx_memory_bytes={MAXIMUM_MLX_MEMORY_BYTES} progressive_expert_payload_bytes={:?} final_expert_payload_bytes={final_expert_payload_bytes} final_active_memory_bytes={final_active_memory_bytes} peak_memory_bytes={peak_memory_bytes} expert_source_read_bytes={expert_source_read_bytes} average_prefill_tokens_per_second={average_prefill_tokens_per_second:.2} average_generation_tokens_per_second={average_generation_tokens_per_second:.2} output_characters={}",
        memory_evidence.progressive_expert_payload_bytes,
        completed_stream.model_text.len(),
    );
}

fn generation_attribution_report_count(isolated_worker_home: &Path) -> usize {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .count()
}

fn decode_streamed_layer_indices(isolated_worker_home: &Path) -> std::collections::BTreeSet<usize> {
    isolated_worker_log_lines(isolated_worker_home)
        .into_iter()
        .filter(|log_line| {
            log_line.contains("Rust expert layer streaming completed")
                && !log_line.contains("streamed_expert_count=256")
        })
        .filter_map(|log_line| {
            log_line
                .split_whitespace()
                .find_map(|field| field.strip_prefix("layer_index="))
                .and_then(|layer_index| layer_index.parse::<usize>().ok())
        })
        .collect()
}

fn memory_admission_decision_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    isolated_worker_log_lines(isolated_worker_home)
        .into_iter()
        .filter(|log_line| log_line.contains("adaptive RAM growth admission decision"))
        .collect()
}

fn retained_expert_fill_decision_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    isolated_worker_log_lines(isolated_worker_home)
        .into_iter()
        .filter(|log_line| {
            log_line.contains("retained expert fill budget decision")
                || log_line.contains("retained expert layer candidate decision")
                || log_line.contains("retained expert layer admitted")
        })
        .collect()
}

fn generation_expert_source_read_bytes(isolated_worker_home: &Path) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the paging acceptance journey should write performance attribution");
    let generation_reports = attribution_log
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each performance-attribution row should be valid JSON")
        })
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .collect::<Vec<_>>();
    assert_eq!(generation_reports.len(), 1);
    // This is logical positional source traffic, not guaranteed physical SSD
    // traffic. macOS process I/O deltas are reported separately because the file
    // cache can satisfy some repeated ranges without another device read.
    generation_reports[0]["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|counter_report| counter_report["counter"] == "positional_file_read_byte_count")
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

fn isolated_worker_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let logging_entries = fs::read_dir(logging_directory)
        .expect("the paging acceptance journey should create its logging directory");
    let mut log_lines = Vec::new();
    for logging_entry in logging_entries {
        let log_path = logging_entry
            .expect("the isolated log entry should be readable")
            .path();
        if log_path.is_file() {
            let log_content = fs::read_to_string(&log_path).unwrap_or_else(|log_read_error| {
                panic!(
                    "{} should be readable: {log_read_error}",
                    log_path.display()
                )
            });
            log_lines.extend(log_content.lines().map(str::to_owned));
        }
    }
    log_lines
}

struct ProgressiveExpertMemoryEvidence {
    progressive_expert_payload_bytes: Vec<u64>,
    final_status: Value,
}

async fn observe_progressive_expert_memory(
    server_address: std::net::SocketAddr,
) -> ProgressiveExpertMemoryEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_prompt_processing = false;
    let mut progressive_expert_payload_bytes = Vec::new();
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_status_progress(&status_document);
            last_status_log_at = Instant::now();
        }
        if status_document["activity"] == "prompt_processing" {
            observed_prompt_processing = true;
            record_expert_payload_increase(&status_document, &mut progressive_expert_payload_bytes);
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_prompt_processing
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return ProgressiveExpertMemoryEvidence {
                progressive_expert_payload_bytes,
                final_status: status_document,
            };
        }
        assert!(Instant::now() < deadline);
        sleep(Duration::from_millis(100)).await;
    }
}

fn log_status_progress(status_document: &Value) {
    let phase = status_document["progress"]["phase"]
        .as_str()
        .unwrap_or("idle");
    let processed_tokens = status_document["progress"]["processed_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total_tokens = status_document["progress"]["total_tokens"]
        .as_u64()
        .unwrap_or(0);
    let elapsed_millis = status_document["progress"]["elapsed_ms"]
        .as_u64()
        .unwrap_or(0);
    let observed_tokens_per_second = if elapsed_millis == 0 {
        0.0
    } else {
        processed_tokens as f64 * 1_000.0 / elapsed_millis as f64
    };
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "[progressive-expert-memory] status=progress phase={phase} processed_tokens={processed_tokens} total_tokens={total_tokens} elapsed_seconds={:.3} observed_tokens_per_second={observed_tokens_per_second:.2} expert_payload_bytes={expert_payload_bytes}",
        elapsed_millis as f64 / 1_000.0,
    );
}

fn record_expert_payload_increase(
    status_document: &Value,
    progressive_expert_payload_bytes: &mut Vec<u64>,
) {
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    let largest_recorded_expert_payload_bytes = progressive_expert_payload_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if expert_payload_bytes > largest_recorded_expert_payload_bytes {
        progressive_expert_payload_bytes.push(expert_payload_bytes);
        eprintln!(
            "[progressive-expert-memory] status=progress processed_tokens={} expert_payload_bytes={expert_payload_bytes}",
            status_document["progress"]["processed_tokens"]
        );
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
        let stream_chunk = stream_item.expect("the public REST stream should remain healthy");
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

fn write_qualification_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the memory-management configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": 23,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        // This cell streams experts. Submit each completed decoder layer so
        // operation-local pages can detach instead of remaining live in a
        // multi-layer lazy tape until the terminal eval.
        "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": false,
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the memory-management configuration should serialize"),
    )
    .expect("the memory-management configuration should be written");
}
