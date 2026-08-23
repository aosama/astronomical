//! User journey proving that reported MLX memory values change during prompt
//! processing, confirming the menu bar's MLX memory bar updates live.
//!
//! The user sees a stacked memory bar in the macOS menu that should reflect
//! evolving memory attribution as the model processes a prompt. If the bar
//! stayed flat, the user would conclude it was frozen or cosmetic. This
//! journey proves otherwise: at least two distinct MLX memory snapshots
//! with different active-memory bytes appear while the model processes
//! prompt tokens from zero through at least 2,000.
//!
//! Ornith 1.5 35B A3B oQ6e MTP is a mixture-of-experts architecture.
//! Its six-bit quantized experts are paged from SSD under a 32 GB ceiling.
//! Each prefill chunk brings new expert pages into MLX memory, making attribution
//! grow visibly chunk by chunk.
//!
//! Acceptance criteria:
//!
//! 1. The REST stream completes with model output (not empty and not an
//!    error).
//! 2. During prompt processing, the status endpoint reports at least two
//!    distinct `active_memory_bytes` values in the `mlx_memory_snapshot`.
//! 3. At least one snapshot is observed with `processed_tokens` >= 2,000
//!    during prompt processing.
//! 4. The snapshot source during prompt processing is `"prefill"`.
//! 5. Peak memory stays within the 32 GB ceiling plus a 1% tolerance.

use std::path::Path;

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
// This ceiling defines a reproducible acceptance cell only. Production code
// must not hardwire it or assume this model always pages experts.
const MAXIMUM_MLX_MEMORY_BYTES: u64 = 32_000_000_000;
const MAXIMUM_MLX_MEMORY_GB: u64 = 32;
const PROMPT_TOKEN_COUNT: usize = 5_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 256;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 128;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const MINIMUM_PROCESSED_TOKENS_FOR_PROGRESS: u64 = 2_000;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST server and real worker to accept MLX memory progress during prompt processing"]
async fn should_report_changing_bounded_mlx_memory_during_prefill() {
    timeout(JOURNEY_TIMEOUT, run_mlx_memory_progress_rest_journey())
        .await
        .expect("the MLX memory progress REST journey must finish within 115 seconds");
}

async fn run_mlx_memory_progress_rest_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the MLX memory progress worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let repeated_source = ROMEO_AND_JULIET_SOURCE.repeat(2);
    let user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &repeated_source,
        "Summarize Romeo and Juliet briefly. Mention the central conflict and tragic outcome.",
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
    let (completed_stream, memory_progress_evidence) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_mlx_memory_progress(server_address),
    );
    // Acceptance criterion 1: the model produces output.
    assert!(
        !completed_stream.model_text.is_empty(),
        "the model must produce output text"
    );
    assert!(matches!(
        completed_stream.finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    // Acceptance criterion 2: at least two distinct active_memory_bytes values
    // appeared during prompt processing. If all snapshots report the same
    // active memory, the bar would not update and the user would see a flat
    // display throughout prefill.
    let distinct_active_memory_values: Vec<u64> = memory_progress_evidence
        .prefill_active_memory_bytes
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    assert!(
        distinct_active_memory_values.len() >= 2,
        "MLX memory should report at least two distinct active_memory_bytes during prompt processing, but found {:?}",
        distinct_active_memory_values
    );
    eprintln!(
        "[mlx-memory-progress] distinct active_memory_bytes during prefill: {:?}",
        distinct_active_memory_values
    );
    // Acceptance criterion 3: at least one snapshot observed during prompt
    // processing shows processed_tokens >= 2,000. This proves the bar
    // updates while the model is actively processing a meaningful portion of
    // the prompt, not just at the start.
    let maximum_observed_prefill_tokens = memory_progress_evidence
        .maximum_prefill_processed_tokens
        .unwrap_or(0);
    assert!(
        maximum_observed_prefill_tokens >= MINIMUM_PROCESSED_TOKENS_FOR_PROGRESS,
        "at least one prefill snapshot should report >= {MINIMUM_PROCESSED_TOKENS_FOR_PROGRESS} processed tokens, but the maximum observed was {maximum_observed_prefill_tokens}"
    );
    // Acceptance criterion 4: the snapshot source during prompt processing is
    // "prefill". This confirms the bar attribution comes from the correct
    // production code path, not an idle sample.
    assert!(
        memory_progress_evidence.observed_prefill_source,
        "at least one MLX memory snapshot during prompt processing must have source \"prefill\""
    );
    // Acceptance criterion 5: peak memory stays within the 32 GB ceiling plus
    // 1% tolerance (for allocator bookkeeping).
    let peak_memory_bytes = memory_progress_evidence
        .peak_active_memory_bytes
        .unwrap_or(0);
    let memory_ceiling_tolerance = (MAXIMUM_MLX_MEMORY_BYTES as f64 * 0.01) as u64;
    assert!(
        peak_memory_bytes <= MAXIMUM_MLX_MEMORY_BYTES.saturating_add(memory_ceiling_tolerance),
        "peak MLX memory {peak_memory_bytes} should stay within the {MAXIMUM_MLX_MEMORY_BYTES} ceiling plus 1% tolerance"
    );
    stop_real_model_rest_server(real_model_rest_server).await;
    eprintln!(
        "[mlx-memory-progress] status=success prompt_tokens={PROMPT_TOKEN_COUNT} \
         maximum_mlx_memory_bytes={MAXIMUM_MLX_MEMORY_BYTES} \
         distinct_active_memory_count={} \
         maximum_prefill_tokens={maximum_observed_prefill_tokens} \
         peak_active_memory_bytes={peak_memory_bytes} \
         output_characters={}",
        distinct_active_memory_values.len(),
        completed_stream.model_text.len(),
    );
}

struct MlxMemoryProgressEvidence {
    /// Active-memory bytes from each prefill snapshot, in observation order.
    prefill_active_memory_bytes: Vec<u64>,
    /// Maximum processed_tokens seen during prompt_processing activity.
    maximum_prefill_processed_tokens: Option<u64>,
    /// Whether at least one snapshot had source "prefill".
    observed_prefill_source: bool,
    /// Highest active_memory_bytes seen across all observations.
    peak_active_memory_bytes: Option<u64>,
}

async fn observe_mlx_memory_progress(
    server_address: std::net::SocketAddr,
) -> MlxMemoryProgressEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_active_request = false;
    let mut prefill_active_memory_bytes: Vec<u64> = Vec::new();
    let mut maximum_prefill_processed_tokens: Option<u64> = None;
    let mut observed_prefill_source = false;
    let mut peak_active_memory_bytes: Option<u64> = None;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            log_status_progress(&status_document);
            last_status_log_at = Instant::now();
        }
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        if activity == "prompt_processing" {
            observed_active_request = true;
            let snapshot_source = status_document["mlx_memory_snapshot"]["source"]
                .as_str()
                .unwrap_or("");
            let active_memory_bytes = status_document["mlx_memory_snapshot"]["active_memory_bytes"]
                .as_u64()
                .unwrap_or(0);
            let processed_tokens = status_document["progress"]["processed_tokens"]
                .as_u64()
                .unwrap_or(0);
            if snapshot_source == "prefill" {
                observed_prefill_source = true;
                prefill_active_memory_bytes.push(active_memory_bytes);
                let current_peak = peak_active_memory_bytes.unwrap_or(0);
                if active_memory_bytes > current_peak {
                    peak_active_memory_bytes = Some(active_memory_bytes);
                }
                let current_max = maximum_prefill_processed_tokens.unwrap_or(0);
                if processed_tokens > current_max {
                    maximum_prefill_processed_tokens = Some(processed_tokens);
                }
            }
        }
        // Also track the decode phase for peak memory.
        if activity == "generating" || activity == "generation" {
            let active_memory_bytes = status_document["mlx_memory_snapshot"]["active_memory_bytes"]
                .as_u64()
                .unwrap_or(0);
            let current_peak = peak_active_memory_bytes.unwrap_or(0);
            if active_memory_bytes > current_peak {
                peak_active_memory_bytes = Some(active_memory_bytes);
            }
        }
        let snapshot_source = status_document["mlx_memory_snapshot"]["source"].as_str();
        if observed_active_request
            && activity == "idle"
            && matches!(snapshot_source, Some("finalized" | "idle_poll"))
        {
            // Capture the final idle snapshot for peak memory comparison.
            let final_active = status_document["mlx_memory_snapshot"]["active_memory_bytes"]
                .as_u64()
                .unwrap_or(0);
            let current_peak = peak_active_memory_bytes.unwrap_or(0);
            if final_active > current_peak {
                peak_active_memory_bytes = Some(final_active);
            }
            // Also capture the final idle peak which may be higher than
            // any live observation.
            let final_peak = status_document["mlx_memory_snapshot"]["peak_memory_bytes"]
                .as_u64()
                .unwrap_or(0);
            if final_peak > peak_active_memory_bytes.unwrap_or(0) {
                peak_active_memory_bytes = Some(final_peak);
            }
            return MlxMemoryProgressEvidence {
                prefill_active_memory_bytes,
                maximum_prefill_processed_tokens,
                observed_prefill_source,
                peak_active_memory_bytes,
            };
        }
        assert!(
            Instant::now() < deadline,
            "the MLX memory progress REST journey did not return to idle: {status_document}"
        );
        sleep(Duration::from_millis(100)).await;
    }
}

fn log_status_progress(status_document: &Value) {
    let activity = status_document["activity"].as_str().unwrap_or("unknown");
    let phase = status_document["progress"]["phase"]
        .as_str()
        .unwrap_or("none");
    let processed_tokens = status_document["progress"]["processed_tokens"]
        .as_u64()
        .unwrap_or(0);
    let total_tokens = status_document["progress"]["total_tokens"]
        .as_u64()
        .unwrap_or(0);
    let active_memory_bytes = status_document["mlx_memory_snapshot"]["active_memory_bytes"]
        .as_u64()
        .unwrap_or(0);
    let snapshot_source = status_document["mlx_memory_snapshot"]["source"]
        .as_str()
        .unwrap_or("none");
    let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
        .as_u64()
        .unwrap_or(0);
    eprintln!(
        "[mlx-memory-progress] status=progress activity={activity} phase={phase} \
         processed_tokens={processed_tokens} total_tokens={total_tokens} \
         snapshot_source={snapshot_source} active_memory_bytes={active_memory_bytes} \
         expert_payload_bytes={expert_payload_bytes}"
    );
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

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    std::fs::create_dir(&configuration_directory)
        .expect("the MLX memory progress configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": MAXIMUM_MLX_MEMORY_GB,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
        },
    });
    std::fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the MLX memory progress configuration should serialize"),
    )
    .expect("the MLX memory progress configuration should be written");
}
