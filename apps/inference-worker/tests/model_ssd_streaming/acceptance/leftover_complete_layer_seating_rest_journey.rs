//! User journey proving leftover complete expert layers stay in RAM after a squeeze.
//!
//! The OptiQ-5bit fixture can hold every expert at 35 GB. This 26 GB cell is just
//! under that complete-owner idle snapshot, so the request cannot keep the atomic
//! complete blob. Leftover RAM can still hold tens of gigabytes of complete layers.
//! Generating with Experts 0.00 GB is the failure this journey exists to catch.
//!
//! Qualification ceilings are evidence cells, not production constants.

use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = "Ornith-1.5-35B-A3B-OptiQ-5bit";
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 26_000_000_000;
const PROMPT_TOKEN_COUNT: usize = 4_096;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 128;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 64;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const REQUEST_MUST_BECOME_ACTIVE_WITHIN: Duration = Duration::from_secs(20);
const JOURNEY_DEADLINE: Duration = Duration::from_secs(115);
// Floor against the 0.00 GB generate failure. Not an exact layer-count golden master.
const MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES: u64 = 1_000_000_000;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_50000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept leftover complete-layer seating after a memory squeeze"]
async fn should_keep_leftover_complete_expert_layers_in_ram_during_squeezed_generation() {
    timeout(
        JOURNEY_DEADLINE,
        run_leftover_complete_layer_seating_rest_journey(),
    )
    .await
    .expect("the leftover complete-layer seating REST journey must finish within 115 seconds");
}

async fn run_leftover_complete_layer_seating_rest_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home = isolated_leftover_seating_worker_home();
    write_acceptance_config(&isolated_worker_home, &model_directory);
    let user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Summarize Romeo and Juliet in one concise paragraph. Include the central conflict, major decisions, and tragic outcome.",
        PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        &isolated_worker_home,
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
        "temperature": 1,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    eprintln!(
        "[leftover-complete-layer-seating] status=progress phase=request_send prompt_characters={} ceiling_bytes={MAXIMUM_MLX_MEMORY_BYTES}",
        user_message.len()
    );
    let streamed_completion: StreamResponse<Value> = timeout(
        Duration::from_secs(60),
        openai_client.chat().create_stream_byot(completion_request),
    )
    .await
    .expect("the squeezed leftover-seating REST request must be accepted within 60 seconds")
    .expect("the leftover-seating REST request should start");
    let (completed_stream, generation_memory) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_generation_expert_payload(server_address),
    );
    stop_real_model_rest_server(real_model_rest_server).await;
    assert!(!completed_stream.model_text.is_empty());
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    assert!(
        generation_memory.saw_generating_activity,
        "the squeezed request must reach token generation: {generation_memory:?}"
    );
    assert!(
        generation_memory.largest_generating_expert_payload_bytes
            >= MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES,
        "leftover RAM that can hold complete layers must not generate with empty expert RAM: {generation_memory:?}"
    );
    eprintln!(
        "[leftover-complete-layer-seating] status=success expert_memory_mode={} largest_generating_expert_payload_bytes={} average_prefill_tokens_per_second={:.2} average_generation_tokens_per_second={:.2} output_characters={}",
        generation_memory.final_expert_memory_mode,
        generation_memory.largest_generating_expert_payload_bytes,
        generation_memory.average_prefill_tokens_per_second,
        generation_memory.average_generation_tokens_per_second,
        completed_stream.model_text.len(),
    );
}

#[derive(Debug)]
struct GenerationMemoryEvidence {
    saw_generating_activity: bool,
    largest_generating_expert_payload_bytes: u64,
    final_expert_memory_mode: String,
    average_prefill_tokens_per_second: f64,
    average_generation_tokens_per_second: f64,
}

async fn observe_generation_expert_payload(server_address: SocketAddr) -> GenerationMemoryEvidence {
    let request_started_at = Instant::now();
    let deadline = request_started_at + JOURNEY_DEADLINE;
    let mut saw_active_request = false;
    let mut saw_generating_activity = false;
    let mut largest_generating_expert_payload_bytes = 0_u64;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    let mut final_status = json!({});
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        let expert_memory_mode = status_document["expert_memory_mode"]
            .as_str()
            .unwrap_or("unavailable");
        let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            eprintln!(
                "[leftover-complete-layer-seating] status=progress activity={activity} expert_memory_mode={expert_memory_mode} expert_payload_bytes={expert_payload_bytes}"
            );
            last_status_log_at = Instant::now();
        }
        if activity != "idle" {
            saw_active_request = true;
        }
        if activity == "generating" {
            saw_generating_activity = true;
            largest_generating_expert_payload_bytes =
                largest_generating_expert_payload_bytes.max(expert_payload_bytes);
            assert!(
                expert_payload_bytes >= MINIMUM_GENERATION_EXPERT_PAYLOAD_BYTES,
                "generation must keep leftover complete layers in RAM, not Experts 0.00 GB: {status_document}"
            );
        }
        if !saw_active_request && request_started_at.elapsed() >= REQUEST_MUST_BECOME_ACTIVE_WITHIN
        {
            panic!(
                "the squeezed leftover-seating request stayed idle for {} seconds: {status_document}",
                REQUEST_MUST_BECOME_ACTIVE_WITHIN.as_secs()
            );
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if saw_active_request
            && activity == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            final_status = status_document;
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the leftover-seating REST journey did not return to idle: {status_document}"
        );
        sleep(Duration::from_millis(100)).await;
    }
    GenerationMemoryEvidence {
        saw_generating_activity,
        largest_generating_expert_payload_bytes,
        final_expert_memory_mode: final_status["expert_memory_mode"]
            .as_str()
            .unwrap_or("unavailable")
            .to_owned(),
        average_prefill_tokens_per_second:
            final_status["serving_session"]["average_prefill_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        average_generation_tokens_per_second:
            final_status["serving_session"]["average_generation_tok_per_second"]
                .as_f64()
                .unwrap_or(0.0),
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
                eprintln!(
                    "[leftover-complete-layer-seating] status=stream_error error={stream_error}"
                );
                break;
            }
        };
        if !stream_chunk["error"].is_null() {
            panic!("the leftover-seating REST stream returned an error: {stream_chunk}");
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

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the leftover-seating configuration directory should be created");
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
            .expect("the leftover-seating configuration should serialize"),
    )
    .expect("the leftover-seating configuration should be written");
}

fn isolated_leftover_seating_worker_home() -> PathBuf {
    let worker_home = std::env::temp_dir().join("astronomical-leftover-complete-layer-seating-e2e");
    let _ = fs::remove_dir_all(&worker_home);
    fs::create_dir_all(&worker_home).expect("the leftover-seating worker home should be created");
    worker_home
        .canonicalize()
        .expect("the leftover-seating worker home should canonicalize")
}
