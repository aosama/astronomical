//! Production journey reproducing a memory setting applied while a generation
//! is active.
//!
//! The user lowers the public "Maximum model RAM" control mid-generation, so
//! the worker queues the ceiling instead of applying it in place. The resident
//! engine cannot shrink below its loaded memory, so after the generation
//! finalizes the daemon must still effectuate the request: replace the
//! worker with a streaming engine under the persisted lower ceiling, clear the
//! pending state and the restart-required error, and keep serving. This is the
//! exact journey that wedged the Development app with a permanently pending
//! ceiling and a rejected retry, and it only reproduces with a real resident
//! model serving real traffic.

use std::{fs, path::Path};

use astronomical_model_serving::{
    CompleteResidencyHeadroomBoundary, MemoryPhase, MlxRamBudget, MlxRamBudgetModelGeometry,
    Qwen3_5ArtifactValidator, mlx_ram_budget_model_geometry_from_validated_artifact,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, Instant, timeout};

use crate::support::openai_client::{ChatCompletionStream, LocalOpenAiClient};
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server,
    put_json_endpoint_expecting_status, stop_real_model_rest_server,
};
use crate::support::{
    artifact_bytes::acceptance_model_scan_root, configured_installed_model_directory_by_id,
    exact_model_prompt, resident_sparse_moe_model_id,
};

const MAXIMUM_REAL_MODEL_MEMORY_BYTES: u64 = 36_000_000_000;
/// The daemon expands every whole-GB ceiling input at 2^30 bytes per GB
/// (the marketed-gigabyte convention); all GB arithmetic in this journey
/// must use the same unit or the pending/effective echoes mismatch the
/// expectations by roughly seven percent.
const BYTES_PER_MARKETED_GIGABYTE: u64 = 1_073_741_824;
const FIXED_PREFILL_CHUNK_SIZE_TOKENS: u32 = 2_048;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 8;
const QUEUED_GENERATION_MAXIMUM_OUTPUT_TOKENS: u32 = 128;
const SMALL_PROMPT_TOKEN_COUNT: usize = 256;
const CONVERGENCE_POLL_INTERVAL: Duration = Duration::from_millis(100);
const CONVERGENCE_PROGRESS_INTERVAL: Duration = Duration::from_secs(5);
/// Bounds the wait for the queued demotion to complete inside the journey.
const CONVERGENCE_DEADLINE: Duration = Duration::from_secs(30);
/// Bounds the wait for the first generation to become observable as active.
const GENERATION_START_DEADLINE: Duration = Duration::from_secs(30);
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the resident engine, queues a lower ceiling behind a live generation, and requires the demotion to converge"]
#[serial]
async fn should_demote_the_live_ceiling_queued_behind_an_active_generation() {
    timeout(SSD_JOURNEY_TIMEOUT, run_queued_demotion_journey())
        .await
        .expect("the queued lower-ceiling demotion SSD journey must finish within 120 seconds");
}

async fn run_queued_demotion_journey() {
    let model_id = resident_sparse_moe_model_id();
    let model_directory = configured_installed_model_directory_by_id(model_id);
    let artifact_payload_bytes = artifact_payload_bytes(&model_directory);
    let residency_geometry = measure_sparse_artifact_residency_geometry(&model_directory);
    // The resident start must purchase the request-time prefill activation
    // workspace on top of the startup residency terms — the same bootstrap
    // RAM budget arithmetic as the resident engine-selection journey; see
    // that journey for the full rationale (issue #1297 / #1508).
    let bootstrap_ram_budget = MlxRamBudget::new(
        residency_geometry.complete_residency_with_headroom_bytes,
        residency_geometry.model_geometry,
    )
    .expect("the resident artifact geometry should compose a bootstrap RAM budget");
    let prefill_activation_workspace_bytes = bootstrap_ram_budget.activation_headroom_bytes(
        MemoryPhase::Prefill,
        u64::from(FIXED_PREFILL_CHUNK_SIZE_TOKENS),
    );
    let resident_memory_ceiling_bytes = residency_geometry
        .complete_residency_with_headroom_bytes
        .saturating_add(BEHAVIOR_PROMPT_CONTEXT_BYTES)
        .saturating_add(prefill_activation_workspace_bytes);
    assert!(
        resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
        "the queued-demotion acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
    );
    let lowered_ceiling_bytes = floor_to_whole_marketed_gigabytes(
        artifact_payload_bytes.saturating_sub(DEMOTION_MARGIN_BELOW_ARTIFACT_BYTES),
    );
    assert!(
        lowered_ceiling_bytes < artifact_payload_bytes,
        "the demotion target must sit below the artifact payload: target={lowered_ceiling_bytes} artifact={artifact_payload_bytes}"
    );

    let worker_home =
        tempfile::tempdir().expect("the queued-demotion worker home should be created");
    write_acceptance_config(
        worker_home.path(),
        &model_directory,
        model_id,
        resident_memory_ceiling_bytes,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory.clone(),
        worker_home.path(),
        resident_memory_ceiling_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;

    let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Explain how haste shapes the tragedy in one concise sentence.",
        SMALL_PROMPT_TOKEN_COUNT,
    );
    let openai_client = LocalOpenAiClient::new(server_address, "local-queued-demotion-client");
    let queued_generation_request = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_content}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": QUEUED_GENERATION_MAXIMUM_OUTPUT_TOKENS
    });
    let mut queued_generation_stream = openai_client
        .create_streaming_chat_completion(&queued_generation_request)
        .await
        .expect("the baseline generation should start before the ceiling is queued");

    wait_for_active_generation(server_address).await;
    let lowered_ceiling_gb = lowered_ceiling_bytes / BYTES_PER_MARKETED_GIGABYTE;
    let queued_ceiling_document = put_json_endpoint_expecting_status(
        server_address,
        "/v1/config/maximum-mlx-memory",
        &json!({"maximum_mlx_memory_gb": lowered_ceiling_gb}),
        202,
    )
    .await;
    assert_eq!(
        queued_ceiling_document["pending_mlx_memory_ceiling_bytes"].as_u64(),
        Some(lowered_ceiling_bytes),
        "the mid-generation ceiling request must be queued, not rejected: {queued_ceiling_document}"
    );
    let conflict_document = put_json_endpoint_expecting_status(
        server_address,
        "/v1/config/maximum-mlx-memory",
        &json!({"maximum_mlx_memory_gb": lowered_ceiling_gb}),
        409,
    )
    .await;
    assert!(
        conflict_document["message"]
            .as_str()
            .is_some_and(|message| message.contains("queued")),
        "a second apply while the queued setting is in flight must be refused: {conflict_document}"
    );

    let queued_generation_response =
        drain_generation_stream(&mut queued_generation_stream, "queued-baseline").await;
    assert!(
        !queued_generation_response.trim().is_empty(),
        "the queued-baseline generation should return visible model output"
    );

    let demoted_status_document =
        wait_for_demotion_convergence(server_address, model_id, lowered_ceiling_bytes).await;
    assert!(
        matches!(
            demoted_status_document["expert_memory_mode"].as_str(),
            Some("paged" | "hybrid")
        ),
        "serving below the artifact size requires streaming execution: {demoted_status_document}"
    );

    serve_and_require_output(
        server_address,
        &openai_client,
        model_id,
        json!([
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": QUEUED_BASELINE_ASSISTANT_PLACEHOLDER},
            {"role": "user", "content": "Now answer again in one concise sentence: what does the play say about haste?"}
        ]),
        "streaming-follow-up",
    )
    .await;

    stop_real_model_rest_server(real_model_rest_server).await;
}

const QUEUED_BASELINE_ASSISTANT_PLACEHOLDER: &str = "(baseline answer)";
/// Context reserve added to the complete-residency ceiling for the baseline prompt.
const BEHAVIOR_PROMPT_CONTEXT_BYTES: u64 = 100_000_000;
/// Gap keeping the demotion target clearly below the artifact payload.
const DEMOTION_MARGIN_BELOW_ARTIFACT_BYTES: u64 = 4_000_000_000;

struct SparseArtifactResidencyGeometry {
    complete_residency_with_headroom_bytes: u64,
    /// The measured geometry the request-time activation reserve composes from.
    model_geometry: MlxRamBudgetModelGeometry,
}

fn measure_sparse_artifact_residency_geometry(
    model_directory: &Path,
) -> SparseArtifactResidencyGeometry {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        mlx_ram_budget_model_geometry_from_validated_artifact(&validated_artifact, model_directory)
            .expect("the configured sparse artifact should expose disk RAM geometry");
    let complete_residency_boundary = CompleteResidencyHeadroomBoundary::from_model_geometry(
        model_geometry,
        required_headroom_bytes,
    );
    SparseArtifactResidencyGeometry {
        complete_residency_with_headroom_bytes: complete_residency_boundary
            .static_complete_residency_bytes
            .saturating_add(complete_residency_boundary.required_headroom_bytes),
        model_geometry,
    }
}

fn artifact_payload_bytes(model_directory: &Path) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    validated_artifact.total_payload_bytes()
}

/// Floors to a whole marketed gigabyte (2^30 bytes): the unit the daemon
/// re-expands whole-GB ceiling inputs at, so the floored target and the
/// daemon's effective echo compare exactly.
fn floor_to_whole_marketed_gigabytes(byte_count: u64) -> u64 {
    (byte_count / BYTES_PER_MARKETED_GIGABYTE) * BYTES_PER_MARKETED_GIGABYTE
}

fn write_acceptance_config(
    worker_home: &Path,
    model_directory: &Path,
    model_id: &str,
    mlx_memory_ceiling_bytes: u64,
) {
    let configuration_directory = worker_home.join(".astronomical-dev");
    fs::create_dir_all(&configuration_directory)
        .expect("the isolated acceptance configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [acceptance_model_scan_root(model_directory)],
            "maximum_mlx_memory_gb": mlx_memory_ceiling_bytes.div_ceil(BYTES_PER_MARKETED_GIGABYTE),
        },
        "prompt_cache": {"enabled": false},
        "diagnostics": {"performance_attribution_enabled": true},
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": FIXED_PREFILL_CHUNK_SIZE_TOKENS
        },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
                },
            },
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the acceptance configuration should serialize"),
    )
    .expect("the acceptance configuration should be written");
}

async fn wait_for_active_generation(server_address: std::net::SocketAddr) {
    let generation_start_deadline = Instant::now() + GENERATION_START_DEADLINE;
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        if status_document["activity"].as_str() != Some("idle") {
            return;
        }
        assert!(
            Instant::now() < generation_start_deadline,
            "the queued-baseline generation never became active: {status_document}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn wait_for_demotion_convergence(
    server_address: std::net::SocketAddr,
    model_id: &str,
    lowered_ceiling_bytes: u64,
) -> Value {
    let convergence_deadline = Instant::now() + CONVERGENCE_DEADLINE;
    let mut last_progress_logged_at = Instant::now();
    let convergence_started_at = Instant::now();
    loop {
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        let effective_ceiling_bytes = status_document["mlx_memory_ceiling_bytes"].as_u64();
        let pending_ceiling_bytes = status_document["pending_mlx_memory_ceiling_bytes"].as_u64();
        let memory_limit_error = status_document["mlx_memory_limit_error"].as_str();
        let restored_model_id = status_document["ready_model_id"].as_str();
        let expert_memory_mode = status_document["expert_memory_mode"].as_str();
        // The demotion is only effectuated once the replacement worker serves
        // the previously selected model under the lowered ceiling. The ceiling
        // itself is acknowledged before the model swap finishes, so ceiling
        // convergence alone is not acceptance.
        let demotion_is_complete = effective_ceiling_bytes == Some(lowered_ceiling_bytes)
            && pending_ceiling_bytes.is_none()
            && memory_limit_error.is_none()
            && restored_model_id == Some(model_id)
            && matches!(expert_memory_mode, Some("paged" | "hybrid"));
        if demotion_is_complete {
            eprintln!(
                "[queued-demotion] converged after {} ms",
                convergence_started_at.elapsed().as_millis()
            );
            return status_document;
        }
        assert!(
            Instant::now() < convergence_deadline,
            "the queued ceiling never fully took effect: effective={effective_ceiling_bytes:?} pending={pending_ceiling_bytes:?} error={memory_limit_error:?} ready_model={restored_model_id:?} mode={expert_memory_mode:?} status={status_document}"
        );
        if last_progress_logged_at.elapsed() >= CONVERGENCE_PROGRESS_INTERVAL {
            eprintln!(
                "[queued-demotion] still waiting to converge; elapsed={} ms; effective={effective_ceiling_bytes:?} pending={pending_ceiling_bytes:?} error={memory_limit_error:?} ready_model={restored_model_id:?} mode={expert_memory_mode:?}",
                convergence_started_at.elapsed().as_millis()
            );
            last_progress_logged_at = Instant::now();
        }
        tokio::time::sleep(CONVERGENCE_POLL_INTERVAL).await;
    }
}

async fn drain_generation_stream(
    streamed_completion: &mut ChatCompletionStream,
    request_label: &str,
) -> String {
    let mut assistant_response = String::new();
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = stream_item.unwrap_or_else(|stream_error| {
            panic!("{request_label} should stream a valid response: {stream_error}")
        });
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(text_fragment) = choice["delta"]["content"].as_str() {
                assistant_response.push_str(text_fragment);
            }
        }
    }
    assistant_response
}

async fn serve_and_require_output(
    server_address: std::net::SocketAddr,
    openai_client: &LocalOpenAiClient,
    model_id: &str,
    messages: Value,
    request_label: &str,
) {
    let completion_request = json!({
        "model": model_id,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": 1
    });
    let mut streamed_completion = match openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
    {
        Ok(streamed_completion) => streamed_completion,
        Err(request_error) => {
            let status_document = get_json_endpoint(server_address, "/v1/status").await;
            panic!(
                "{request_label} should start: {request_error}; worker_status={status_document}"
            );
        }
    };
    let assistant_response = drain_generation_stream(&mut streamed_completion, request_label).await;
    assert!(
        !assistant_response.trim().is_empty(),
        "{request_label} should return visible model output"
    );
}
