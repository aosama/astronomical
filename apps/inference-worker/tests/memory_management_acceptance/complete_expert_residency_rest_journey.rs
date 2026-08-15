//! User journey proving that a fitting model stays completely resident.
//!
//! At a 25 GB decimal-SI MLX ceiling, the selected fixture model has enough room
//! for model core, every expert, and required request headroom. The journey sends
//! a long public streaming request and proves three user-visible consequences:
//!
//! - generation completes through the production REST/server/worker stack;
//! - final status truthfully reports complete expert residency;
//! - the generation attribution report contains zero expert positional-read bytes.
//!
//! It also protects the recovery-policy correction: a recovery-only projection
//! shortfall must be admitted when stable and expected peak fit. Reintroducing
//! preemptive recovery eviction would turn this resident journey back into paging.

use std::{fs, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
// Qualification cells are explicit evidence, not production constants. Runtime
// policy continues to derive capacity from user/machine ceiling and model geometry.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 25_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 7_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 1_280;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 256;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept complete expert residency without request-time SSD streaming"]
async fn should_keep_all_experts_resident_and_avoid_expert_ssd_streaming_under_twenty_five_gb() {
    timeout(
        JOURNEY_TIMEOUT,
        run_complete_expert_residency_rest_journey(),
    )
    .await
    .expect("the complete expert-residency REST journey must finish within 115 seconds");
}

async fn run_complete_expert_residency_rest_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the complete-residency worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(3);
    // The prompt builder tokenizes with the real model tokenizer and produces the
    // exact required count. Romeo and Juliet is the repository's required LLM
    // fixture, so this remains representative and reproducible.
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
            .with_api_key("local-acceptance-client"),
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
        .expect("the complete-residency REST request should start");
    let (completed_stream, final_status) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_resident_request_until_idle(server_address),
    );
    // Consume the Server-Sent Events stream and status polling concurrently. This
    // observes active residency while a real client receives output rather than
    // inspecting only a post-request model object.
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
    // Logs are read only after worker shutdown so buffered attribution and tracing
    // have reached their isolated files.
    let memory_admission_decisions =
        memory_admission_decision_log_lines(isolated_worker_home.path());
    for memory_admission_decision in &memory_admission_decisions {
        eprintln!("[complete-expert-residency] status=memory_decision {memory_admission_decision}");
    }
    assert!(
        memory_admission_decisions
            .iter()
            .any(|memory_admission_decision| {
                memory_admission_decision.contains("decision=\"admit_with_recovery_constraint\"")
                    && memory_admission_decision.contains("recovery_reserve_only_trigger=true")
            }),
        "the 25 GB journey must prove that a recovery-only shortfall was admitted: {memory_admission_decisions:?}"
    );
    assert_eq!(
        final_status["expert_memory_mode"].as_str(),
        Some("resident"),
        "the completed request must leave all experts resident; memory_admission_decisions={memory_admission_decisions:?}; final_status={final_status}"
    );
    let expert_source_read_bytes = generation_expert_source_read_bytes(isolated_worker_home.path());
    assert_eq!(
        expert_source_read_bytes, 0,
        "a completely resident model must not stream expert ranges from SSD during the request"
    );
    eprintln!(
        "[complete-expert-residency] status=success prompt_tokens={PROMPT_TOKEN_COUNT} expert_memory_mode=resident expert_source_read_bytes={expert_source_read_bytes} average_prefill_tokens_per_second={average_prefill_tokens_per_second:.2} average_generation_tokens_per_second={average_generation_tokens_per_second:.2} output_characters={}",
        completed_stream.model_text.len(),
    );
}

fn memory_admission_decision_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical-dev").join("logs");
    let logging_entries = fs::read_dir(&logging_directory)
        .expect("the acceptance journey should create its isolated logging directory");
    let mut memory_admission_decisions = Vec::new();
    for logging_entry in logging_entries {
        let logging_entry = logging_entry.expect("the isolated log entry should be readable");
        let log_path = logging_entry.path();
        if !log_path.is_file() {
            continue;
        }
        let log_content = fs::read_to_string(&log_path).unwrap_or_else(|log_read_error| {
            panic!(
                "{} should be readable: {log_read_error}",
                log_path.display()
            )
        });
        memory_admission_decisions.extend(
            log_content
                .lines()
                .filter(|log_line| log_line.contains("adaptive RAM growth admission decision"))
                .map(str::to_owned),
        );
    }
    assert!(
        !memory_admission_decisions.is_empty(),
        "the acceptance journey should capture adaptive memory admission decisions"
    );
    memory_admission_decisions
}

async fn observe_resident_request_until_idle(server_address: std::net::SocketAddr) -> Value {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
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
        "maximum_mlx_memory_gb": 25,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "prompt_processing_chunk_size_optimizer_enabled": false,
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
