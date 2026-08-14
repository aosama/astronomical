//! Real user journey: after paging, generation reclaims granted expert RAM.
//!
//! # What the user does
//!
//! 1. Launch a worker whose public ceiling (26 GB) is just large enough for
//!    complete Ornith 4-bit expert residency when idle.
//! 2. Send one long Romeo and Juliet prompt plus a processed image.
//! 3. Wait for streamed tokens, then look at idle `/v1/status`.
//!
//! # What must happen inside the worker
//!
//! - The model starts `resident`: every expert sits in RAM.
//! - Prefill activations do not fit, so request pressure demotes the complete
//!   owner to `paged` and may freeze retained pages.
//! - After the last prefill barrier, decode handoff lifts that freeze and
//!   either restores complete residency or fills demand-selected pages from
//!   the leftover composed budget.
//! - During `generating`, expert payload must grow well above one routed
//!   top-K working set (~1.07 GB). Crossing 4 GB is the user-visible proof
//!   that the freeze did not stay in place.
//! - After the request, idle promotion must restore `resident` without the
//!   user raising the ceiling. 25 GB was measured too tight for that idle
//!   admit; 26 GB is the focused cell for this restore.
//!
//! This test launches a real REST server, supervisor, worker, model artifact,
//! and MLX runtime. It is not a mock.

use std::{fs, net::SocketAddr, path::Path};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, Instant, sleep, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server, stop_real_model_rest_server,
};

const MODEL_ID: &str = "Ornith-1.0-35B-OptiQ-4bit";
// 25 GB demotes during this prompt but idle complete residency did not fit.
// 26 GB is the focused cell that proves decode reclaim and idle restore.
const MAXIMUM_MLX_MEMORY_GB: u64 = 26;
const PRESSURE_PROMPT_TOKEN_COUNT: usize = 32_000;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 16;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 8;
const PRESSURE_IMAGE_WIDTH: u32 = 980;
const PRESSURE_IMAGE_HEIGHT: u32 = 980;
// Larger than one routed top-K page per layer (~1.07 GB). Generation has
// reclaimed RAM only when occupancy crosses this user-visible floor.
const MINIMUM_RECLAIMED_EXPERT_PAYLOAD_BYTES: u64 = 4_000_000_000;
const STATUS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the real REST server, worker, Ornith model, and MLX runtime to prove decode reclaims expert RAM after paging"]
async fn should_grow_expert_ram_for_generation_after_a_pressure_page_then_restore_residency_when_the_ceiling_still_fits()
 {
    timeout(JOURNEY_TIMEOUT, run_decode_reclaims_expert_ram_journey())
        .await
        .expect("the decode RAM-reclaim journey must finish within 115 seconds");
}

async fn run_decode_reclaims_expert_ram_journey() {
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the decode RAM-reclaim worker home should be created");
    write_acceptance_config(isolated_worker_home.path(), &model_directory);
    let pressure_prompt = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        &ROMEO_AND_JULIET_SOURCE.repeat(12),
        "Summarize Romeo and Juliet, preserving the major decisions and their consequences.",
        PRESSURE_PROMPT_TOKEN_COUNT,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        isolated_worker_home.path(),
        MAXIMUM_MLX_MEMORY_GB * 1_000_000_000,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-decode-ram-reclaim-client"),
    );
    let pressure_image_data_url = crate::common::solid_png::solid_rgb_png_data_url(
        PRESSURE_IMAGE_WIDTH,
        PRESSURE_IMAGE_HEIGHT,
        180,
        32,
        42,
    );
    let streamed_completion =
        start_streaming_request(&openai_client, pressure_prompt, pressure_image_data_url).await;
    let (completion, memory_evidence) = tokio::join!(
        consume_completed_stream(streamed_completion),
        observe_paging_then_reclaimed_generation(server_address),
    );
    let status_after_request = get_json_endpoint(server_address, "/v1/status").await;
    if completion.is_err()
        || status_after_request["status"] == "unavailable"
        || status_after_request["expert_memory_mode"].is_null()
    {
        dump_reclaim_logs(isolated_worker_home.path());
        panic!("pressure request died; status={status_after_request}; stream={completion:?}");
    }
    let completion = completion.expect("the pressure request stream should complete");
    assert!(
        !completion.model_text.is_empty(),
        "the request should return text"
    );
    assert!(
        matches!(completion.finish_reason.as_deref(), Some("stop" | "length")),
        "the request should finish cleanly: {:?}",
        completion.finish_reason
    );
    assert!(
        memory_evidence.observed_resident,
        "the model must start resident"
    );
    assert!(
        memory_evidence.observed_paged,
        "the long request must force paging during prefill"
    );
    if !memory_evidence.reclaimed_expert_payload_during_generation
        && !memory_evidence.observed_resident_generation
    {
        dump_reclaim_logs(isolated_worker_home.path());
        panic!(
            "generation must reclaim expert RAM or return to complete residency; maximum_generation_expert_payload_bytes={}",
            memory_evidence.maximum_generation_expert_payload_bytes
        );
    }
    if status_after_request["expert_memory_mode"].as_str() != Some("resident") {
        dump_reclaim_logs(isolated_worker_home.path());
        panic!(
            "idle promotion must restore complete residency when the original ceiling still fits: {status_after_request}"
        );
    }

    stop_real_model_rest_server(real_model_rest_server).await;
    let worker_logs = isolated_worker_log_lines(isolated_worker_home.path());
    assert!(
        worker_logs.iter().any(|line| {
            line.contains("demoted complete resident experts to Rust streaming")
                && line.contains("transition_reason=RequestPressure")
        }),
        "the journey must prove request-pressure demotion"
    );
    assert!(
        !worker_logs
            .iter()
            .any(|line| line.contains("capped decode-warm fill to a routed working set")),
        "decode-warm must not keep the 1 GB working-set cap after prefill"
    );
    assert!(
        worker_logs.iter().any(|line| {
            line.contains("completed complete-model expert residency admission")
                && line.contains("outcome=\"promoted\"")
                && (line.contains("transition_reason=DecodeHandoff")
                    || line.contains("transition_reason=RequestFinalization"))
        }),
        "decode handoff or idle finalization must promote when the ceiling still fits"
    );
    eprintln!(
        "[decode-ram-reclaim] status=success maximum_generation_expert_payload_bytes={} output_characters={}",
        memory_evidence.maximum_generation_expert_payload_bytes,
        completion.model_text.len(),
    );
}

struct DecodeRamReclaimEvidence {
    observed_resident: bool,
    observed_paged: bool,
    observed_resident_generation: bool,
    reclaimed_expert_payload_during_generation: bool,
    maximum_generation_expert_payload_bytes: u64,
}

async fn observe_paging_then_reclaimed_generation(
    server_address: SocketAddr,
) -> DecodeRamReclaimEvidence {
    let deadline = Instant::now() + JOURNEY_TIMEOUT;
    let mut observed_active_request = false;
    let mut observed_resident = false;
    let mut observed_paged = false;
    let mut observed_resident_generation = false;
    let mut reclaimed_expert_payload_during_generation = false;
    let mut maximum_generation_expert_payload_bytes = 0;
    let mut last_status_log_at = Instant::now() - STATUS_LOG_INTERVAL;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let activity = status_document["activity"].as_str().unwrap_or("unknown");
        let expert_memory_mode = status_document["expert_memory_mode"].as_str();
        let expert_payload_bytes = status_document["mlx_memory_snapshot"]["expert_payload_bytes"]
            .as_u64()
            .unwrap_or(0);
        if activity != "idle" {
            observed_active_request = true;
            observed_resident |= expert_memory_mode == Some("resident");
            observed_paged |= expert_memory_mode == Some("paged");
        }
        if activity == "generating" {
            observed_resident_generation |= expert_memory_mode == Some("resident");
            reclaimed_expert_payload_during_generation |=
                expert_payload_bytes >= MINIMUM_RECLAIMED_EXPERT_PAYLOAD_BYTES;
            maximum_generation_expert_payload_bytes =
                maximum_generation_expert_payload_bytes.max(expert_payload_bytes);
        }
        if last_status_log_at.elapsed() >= STATUS_LOG_INTERVAL {
            eprintln!(
                "[decode-ram-reclaim] status=progress activity={activity} expert_memory_mode={} expert_payload_bytes={expert_payload_bytes} processed_tokens={} total_tokens={}",
                expert_memory_mode.unwrap_or("unavailable"),
                status_document["progress"]["processed_tokens"],
                status_document["progress"]["total_tokens"],
            );
            last_status_log_at = Instant::now();
        }
        if status_document["status"] == "unavailable"
            || (observed_active_request && activity == "idle")
        {
            return DecodeRamReclaimEvidence {
                observed_resident,
                observed_paged,
                observed_resident_generation,
                reclaimed_expert_payload_during_generation,
                maximum_generation_expert_payload_bytes,
            };
        }
        assert!(
            Instant::now() < deadline,
            "the pressure request did not return to idle"
        );
        sleep(Duration::from_millis(50)).await;
    }
}

async fn start_streaming_request(
    openai_client: &Client<OpenAIConfig>,
    user_message: String,
    image_data_url: String,
) -> StreamResponse<Value> {
    let completion_request = json!({
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    openai_client
        .chat()
        .create_stream_byot(completion_request)
        .await
        .expect("the decode RAM-reclaim request should start")
}

#[derive(Debug)]
struct CompletedStream {
    model_text: String,
    finish_reason: Option<String>,
}

async fn consume_completed_stream(
    mut streamed_completion: StreamResponse<Value>,
) -> Result<CompletedStream, String> {
    let mut model_text = String::new();
    let mut finish_reason = None;
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.map_err(|stream_error| stream_error.to_string())?;
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                model_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finish_reason = Some(reason.to_owned());
            }
        }
    }
    Ok(CompletedStream {
        model_text: model_text.trim().to_owned(),
        finish_reason,
    })
}

fn dump_reclaim_logs(isolated_worker_home: &Path) {
    for log_line in isolated_worker_log_lines(isolated_worker_home) {
        if log_line.contains("demoted complete resident")
            || log_line.contains("capped decode-warm")
            || log_line.contains("decode-warm expert pages ready")
            || log_line.contains("starting first decode forward")
            || log_line.contains("DecodeHandoff")
            || log_line.contains("RequestFinalization")
            || log_line.contains("MLX inference owner stopped")
        {
            eprintln!("[decode-ram-reclaim] status=handoff_log {log_line}");
        }
    }
}

fn isolated_worker_log_lines(isolated_worker_home: &Path) -> Vec<String> {
    let logging_directory = isolated_worker_home.join(".astronomical").join("logs");
    fs::read_dir(logging_directory)
        .expect("the decode RAM-reclaim log directory should exist")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .flat_map(|entry| {
            fs::read_to_string(entry.path())
                .unwrap_or_default()
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn write_acceptance_config(isolated_worker_home: &Path, model_directory: &Path) {
    let configuration_directory = isolated_worker_home.join(".astronomical");
    fs::create_dir(&configuration_directory)
        .expect("the decode RAM-reclaim configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [model_directory],
        "maximum_mlx_memory_gb": MAXIMUM_MLX_MEMORY_GB,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": false,
        "performance_attribution_enabled": true,
        "mtp_enabled": false,
        "logging": {"level": "debug", "retained_files": 2},
        "chunking": {
            "prefill_size_optimizer_enabled": false,
            "fixed_prefill_tokens": 2_048,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the decode RAM-reclaim configuration should serialize"),
    )
    .expect("the decode RAM-reclaim configuration should be written");
}
