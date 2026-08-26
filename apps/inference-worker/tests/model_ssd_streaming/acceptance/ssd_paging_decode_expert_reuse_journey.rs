//! User journey proving SSD-paged expert reuse across decode tokens under a constrained 23 GB ceiling.
//!
//! The model cannot keep every expert resident in this qualification cell. The
//! desired behavior is therefore not merely "request succeeds": routed experts
//! must remain reusable across decoder layers instead of consuming nearly the
//! whole ceiling on early layers while repeatedly reading omitted routes.
//!
//! Acceptance criteria (what "good" looks like for the user):
//!
//! 1. The model produces a complete response (non-empty text, finish reason
//!    "stop" or "length").
//! 2. The model is in paging mode: expert payload is non-zero during decode,
//!    confirming the test setup exercises SSD streaming.
//! 3. Memory stays within the configured ceiling (plus a small tolerance).
//! 4. Attribution reports retained-route hits, directly proving that decode
//!    reused expert ownership instead of merely completing through repeated reads.
//! 5. Decode throughput is reported as positive finite evidence without imposing
//!    one laptop's hardware-specific performance threshold.
//! 6. Exactly one generation attribution report is written, proving clean request
//!    completion.

use std::{fs, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
// This ceiling defines a reproducible acceptance cell only. Production code must
// not hardwire it or assume this model always leaves exactly four layers cold.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 23_000_000_000;
// One percent of the ceiling covers allocator rounding and transient peaks that
// settle before the finalized snapshot.
const MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES: u64 = MAXIMUM_MLX_MEMORY_BYTES / 100;
const PROMPT_TOKEN_COUNT: usize = 7_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 10_000;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 1_000;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept expert-memory management behavior"]
async fn should_reuse_retained_decode_experts_while_staying_within_the_mlx_memory_ceiling() {
    timeout(
        JOURNEY_TIMEOUT,
        run_ssd_paging_decode_expert_reuse_journey(),
    )
    .await
    .expect("the progressive expert-memory REST journey must finish within 115 seconds");
}

async fn run_ssd_paging_decode_expert_reuse_journey() {
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
        observe_ssd_paging_decode_expert_reuse(server_address),
    );

    // --- Structural assertion 1: generation completes with real output ---
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));

    // --- Structural assertion 2: the model is in paging mode (some experts
    // retained, not all-zero, confirming the test exercises SSD streaming) ---
    let final_expert_payload_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
    assert!(
        final_expert_payload_bytes > 0,
        "the paged model should retain some expert payload in memory"
    );

    // --- Structural assertion 3: memory stays within the configured ceiling ---
    let final_active_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["active_memory_bytes"]
            .as_u64()
            .expect("the completed status should report final active MLX memory");
    let peak_memory_bytes =
        memory_evidence.final_status["mlx_memory_snapshot"]["peak_memory_bytes"]
            .as_u64()
            .expect("the completed status should report peak MLX memory");
    assert!(
        final_active_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES,
        "final active memory {final_active_memory_bytes} must stay within ceiling {MAXIMUM_MLX_MEMORY_BYTES}"
    );
    assert!(
        peak_memory_bytes
            <= MAXIMUM_MLX_MEMORY_BYTES.saturating_add(MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES),
        "peak memory {peak_memory_bytes} must stay within ceiling plus tolerance {}",
        MAXIMUM_MLX_MEMORY_BYTES.saturating_add(MAXIMUM_MLX_MEMORY_TOLERANCE_BYTES)
    );

    // --- Structural assertion 4: decode reused retained route assignments ---
    stop_real_model_rest_server(real_model_rest_server).await;
    let retained_route_assignment_hit_count = generation_attribution_counter(
        isolated_worker_home.path(),
        "retained_route_assignment_hit_count",
    );
    assert!(
        retained_route_assignment_hit_count > 0,
        "decode must reuse at least one retained expert route assignment"
    );

    // --- Measured assertion 5: throughput remains portable evidence ---
    let average_generation_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_generation_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average generation throughput");
    let average_prefill_tokens_per_second =
        memory_evidence.final_status["serving_session"]["average_prefill_tok_per_second"]
            .as_f64()
            .expect("the completed status should report average prefill throughput");
    assert!(
        average_generation_tokens_per_second.is_finite()
            && average_generation_tokens_per_second > 0.0,
        "decode throughput must be a positive finite measurement"
    );

    // --- Structural assertion 6: exactly one generation attribution report ---
    assert_eq!(
        generation_attribution_report_count(isolated_worker_home.path()),
        1,
        "exactly one generation attribution report should be written"
    );

    // Diagnostic output: useful for debugging but not asserted.
    let expert_source_read_bytes = generation_expert_source_read_bytes(isolated_worker_home.path());
    let decode_streamed_layer_indices = decode_streamed_layer_indices(isolated_worker_home.path());
    let retained_expert_payload_increments = memory_evidence.retained_expert_payload_bytes.len();
    eprintln!(
        "[ssd-paging-decode-expert-reuse] status=success \
         prompt_tokens={PROMPT_TOKEN_COUNT} \
         maximum_mlx_memory_gb={} \
         final_expert_payload_gb={:.2} \
         final_active_memory_gb={:.2} \
         peak_memory_gb={:.2} \
         expert_source_read_gb={:.2} \
         retained_route_hits={retained_route_assignment_hit_count} \
         decode_streamed_layer_count={} \
         retained_payload_increments={} \
         average_prefill_tok_per_second={average_prefill_tokens_per_second:.2} \
         average_generation_tok_per_second={average_generation_tokens_per_second:.2} \
         output_characters={}",
        MAXIMUM_MLX_MEMORY_BYTES / 1_000_000_000,
        final_expert_payload_bytes as f64 / 1e9,
        final_active_memory_bytes as f64 / 1e9,
        peak_memory_bytes as f64 / 1e9,
        expert_source_read_bytes as f64 / 1e9,
        decode_streamed_layer_indices.len(),
        retained_expert_payload_increments,
        completed_stream.model_text.len(),
    );
}

fn generation_attribution_counter(isolated_worker_home: &Path, counter_identifier: &str) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    fs::read_to_string(attribution_log_path)
        .expect("the completed request should flush performance attribution")
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .flat_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .cloned()
                .unwrap_or_default()
        })
        .filter(|counter_report| counter_report["counter"] == counter_identifier)
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
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

fn generation_expert_source_read_bytes(isolated_worker_home: &Path) -> u64 {
    let attribution_log_path = isolated_worker_home
        .join(".astronomical-dev")
        .join("logs")
        .join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(attribution_log_path)
        .expect("the paging acceptance journey should write performance attribution");
    attribution_log
        .lines()
        .filter_map(|json_line| serde_json::from_str::<Value>(json_line).ok())
        .filter(|attribution_report| attribution_report["report_kind"] == "generation")
        .filter_map(|attribution_report| {
            attribution_report["counters"]
                .as_array()
                .map(|counters| counters.to_owned())
        })
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
    retained_expert_payload_bytes: Vec<u64>,
    final_status: Value,
}

async fn observe_ssd_paging_decode_expert_reuse(
    server_address: std::net::SocketAddr,
) -> ProgressiveExpertMemoryEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_prompt_processing = false;
    let mut retained_expert_payload_bytes = Vec::new();
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_status_progress(&status_document);
            last_status_log_at = Instant::now();
        }
        if status_document["activity"] == "prompt_processing" {
            observed_prompt_processing = true;
            record_expert_payload_increase(&status_document, &mut retained_expert_payload_bytes);
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_prompt_processing
            && status_document["activity"] == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            return ProgressiveExpertMemoryEvidence {
                retained_expert_payload_bytes,
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
        "[ssd-paging-decode-expert-reuse] status=progress phase={phase} processed_tokens={processed_tokens} total_tokens={total_tokens} elapsed_seconds={:.3} observed_tokens_per_second={observed_tokens_per_second:.2} expert_payload_bytes={expert_payload_bytes}",
        elapsed_millis as f64 / 1_000.0,
    );
}

fn record_expert_payload_increase(
    status_document: &Value,
    retained_expert_payload_bytes: &mut Vec<u64>,
) {
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    let largest_recorded_expert_payload_bytes = retained_expert_payload_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    if expert_payload_bytes > largest_recorded_expert_payload_bytes {
        retained_expert_payload_bytes.push(expert_payload_bytes);
        eprintln!(
            "[ssd-paging-decode-expert-reuse] status=progress processed_tokens={} expert_payload_bytes={expert_payload_bytes}",
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
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        // This cell streams experts. Submit each completed decoder layer so
        // operation-local pages can detach instead of remaining live in a
        // multi-layer lazy tape until the terminal eval.
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2_048,
            "prefill_graph_submission_layer_interval": 0,
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
