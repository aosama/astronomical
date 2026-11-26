//! Production journey proving a live ceiling below the artifact size still serves.
//!
//! The worker loads the sparse MoE at a ceiling that holds the whole model,
//! then the user lowers the public MLX ceiling far below the artifact payload.
//! The one text engine adopts the lower ceiling in place: the same worker
//! process keeps the model loaded, streams from the SSD what no longer fits,
//! and keeps serving the same conversation. This is the user-facing behavior
//! behind the Settings "Maximum model RAM" control.

use super::automatic_engine_selection_rest::prompt_chunk_mlx_working_buffer_allowance_bytes;
use std::{fs, path::Path};

use futures_util::StreamExt;
use model_serving::{
    CompleteResidencyHeadroomBoundary, ModelMemoryGeometry, Qwen3_5ArtifactValidator,
    model_memory_geometry_from_validated_artifact,
};
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    RealModelRestServer, SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server,
    put_json_endpoint, stop_real_model_rest_server,
};
use crate::support::worker_process_observation::{
    WorkerProcessTimeline, is_process_running, sole_worker_child_process_identifier,
};
use crate::support::{
    artifact_bytes::acceptance_model_scan_root, configured_installed_model_directory_by_id,
    exact_model_prompt, resident_sparse_moe_model_id,
};

const MAXIMUM_REAL_MODEL_MEMORY_BYTES: u64 = 36_000_000_000;
const FIXED_PREFILL_CHUNK_SIZE_TOKENS: u32 = 2_048;
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 8;
const SMALL_PROMPT_TOKEN_COUNT: usize = 256;
/// Context the whole-model phase reserves: the small prompt, its chat template,
/// and the generated tokens, rounded up so template growth cannot overflow it.
const RESIDENT_PHASE_CONTEXT_TOKEN_COUNT: usize = 512;
use crate::support::marketed_gigabytes::BYTES_PER_MARKETED_GIGABYTE;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the whole model, lowers the live ceiling below the artifact size, and requires the same worker to keep serving by streaming"]
#[serial]
async fn should_serve_after_lowering_the_live_ceiling_below_the_artifact_size() {
    timeout(SSD_JOURNEY_TIMEOUT, run_lower_ceiling_journey())
        .await
        .expect("the lower-ceiling SSD journey must finish within 120 seconds");
}

async fn run_lower_ceiling_journey() {
    let ceiling_journey = start_resident_ceiling_journey().await;
    lower_ceiling_and_require_streaming_execution(&ceiling_journey).await;
    stop_real_model_rest_server(ceiling_journey.real_model_rest_server).await;
}

/// The whole-model baseline the live-ceiling journey starts from, plus the
/// measured ceiling the later phase moves to.
struct ResidentCeilingJourney {
    real_model_rest_server: RealModelRestServer,
    // Held for the journey's lifetime so the isolated worker home outlives
    // every worker the daemon launches into it.
    _worker_home: tempfile::TempDir,
    model_id: &'static str,
    prompt_content: String,
    openai_client: LocalOpenAiClient,
    lowered_ceiling_gb: u64,
    lowered_ceiling_bytes: u64,
    initial_floor_bytes: u64,
}

async fn start_resident_ceiling_journey() -> ResidentCeilingJourney {
    let model_id = resident_sparse_moe_model_id();
    let model_directory = configured_installed_model_directory_by_id(model_id);
    let artifact_payload_bytes = artifact_payload_bytes(&model_directory);
    let residency_geometry = measure_sparse_artifact_residency_geometry(&model_directory);
    // The ceiling purchases complete residency, the resident phases' context,
    // and an allowance for one prompt chunk's MLX working buffers, so every
    // request fits and none streams from the SSD on its own.
    let resident_memory_ceiling_bytes = residency_geometry
        .complete_residency_with_headroom_bytes
        .saturating_add(context_memory_reservation_bytes(
            &model_directory,
            RESIDENT_PHASE_CONTEXT_TOKEN_COUNT,
        ))
        .saturating_add(prompt_chunk_mlx_working_buffer_allowance_bytes(
            &residency_geometry.model_geometry,
        ));
    assert!(
        resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
        "the lowered-ceiling acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
    );
    // The lowered target sits well below the artifact payload so only SSD
    // streaming can serve the model, and above the geometry floor so the
    // engine can honestly honor it.
    let lowered_ceiling_bytes =
        artifact_payload_bytes.saturating_sub(LOWERED_MARGIN_BELOW_ARTIFACT_BYTES);
    let lowered_ceiling_gb = lowered_ceiling_bytes / BYTES_PER_MARKETED_GIGABYTE;
    let lowered_ceiling_bytes = lowered_ceiling_gb * BYTES_PER_MARKETED_GIGABYTE;
    assert!(
        lowered_ceiling_bytes < artifact_payload_bytes,
        "the lowered target must sit below the artifact payload: target={lowered_ceiling_bytes} artifact={artifact_payload_bytes}"
    );

    let worker_home =
        tempfile::tempdir().expect("the lowered-ceiling worker home should be created");
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
    let openai_client = LocalOpenAiClient::new(server_address, "local-lowered-ceiling-client");
    serve_and_require_output(
        server_address,
        &openai_client,
        model_id,
        json!([{"role": "user", "content": prompt_content}]),
        "resident-baseline",
    )
    .await;

    let initial_status_document = get_json_endpoint(server_address, "/v1/status").await;
    assert_eq!(
        initial_status_document["expert_memory_mode"], "resident",
        "the journey must start with the whole model held in memory: {initial_status_document}"
    );
    let initial_floor_bytes = initial_status_document["minimum_mlx_memory_ceiling_bytes"]
        .as_u64()
        .expect("the loaded model's status should publish its geometry floor");
    assert!(
        initial_floor_bytes < artifact_payload_bytes,
        "the geometry floor must describe streaming needs, not the produced residency: floor={initial_floor_bytes} artifact={artifact_payload_bytes}"
    );
    assert!(
        lowered_ceiling_bytes >= initial_floor_bytes,
        "the lowered target must remain at or above the streaming geometry floor: target={lowered_ceiling_bytes} floor={initial_floor_bytes}"
    );

    ResidentCeilingJourney {
        real_model_rest_server,
        _worker_home: worker_home,
        model_id,
        prompt_content,
        openai_client,
        lowered_ceiling_gb,
        lowered_ceiling_bytes,
        initial_floor_bytes,
    }
}

async fn lower_ceiling_and_require_streaming_execution(ceiling_journey: &ResidentCeilingJourney) {
    let server_address = ceiling_journey.real_model_rest_server.server_address;
    let lowered_ceiling_gb = ceiling_journey.lowered_ceiling_gb;
    let lowered_ceiling_bytes = ceiling_journey.lowered_ceiling_bytes;
    let initial_floor_bytes = ceiling_journey.initial_floor_bytes;
    let openai_client = &ceiling_journey.openai_client;
    let model_id = ceiling_journey.model_id;
    let prompt_content = &ceiling_journey.prompt_content;
    let original_worker_process_identifier = sole_worker_child_process_identifier(
        "the whole-model baseline must own exactly one real worker process",
    );

    // The engine adopts the ceiling in place, so the worker process must not
    // change; observe that from outside the daemon, which cannot restate it.
    let worker_process_timeline = WorkerProcessTimeline::start();
    let lowered_ceiling_document = put_json_endpoint(
        server_address,
        "/v1/config/maximum-mlx-memory",
        &json!({"maximum_mlx_memory_gb": lowered_ceiling_gb}),
    )
    .await;
    let observed_worker_processes = worker_process_timeline.stop().await;
    assert_worker_process_kept(
        original_worker_process_identifier,
        &observed_worker_processes,
    );
    assert_eq!(
        lowered_ceiling_document["effective_mlx_memory_ceiling_bytes"].as_u64(),
        Some(lowered_ceiling_bytes),
        "the worker must apply the requested ceiling in place: {lowered_ceiling_document}"
    );
    assert_eq!(
        lowered_ceiling_document["minimum_mlx_memory_ceiling_bytes"].as_u64(),
        Some(initial_floor_bytes),
        "the geometry floor must remain stable across the lowered ceiling: {lowered_ceiling_document}"
    );

    let lowered_status_document = get_json_endpoint(server_address, "/v1/status").await;
    assert!(
        matches!(
            lowered_status_document["expert_memory_mode"].as_str(),
            Some("paged" | "hybrid")
        ),
        "serving below the artifact size requires streaming from the SSD: {lowered_status_document}"
    );

    serve_and_require_output(
        server_address,
        openai_client,
        model_id,
        json!([
            {"role": "user", "content": prompt_content},
            {"role": "assistant", "content": BASELINE_ASSISTANT_PLACEHOLDER},
            {"role": "user", "content": "Now answer again in one concise sentence: what does the play say about haste?"}
        ]),
        "streaming-after-lowering",
    )
    .await;
}

const BASELINE_ASSISTANT_PLACEHOLDER: &str = "(baseline answer)";
/// Gap keeping the lowered target clearly below the artifact payload.
const LOWERED_MARGIN_BELOW_ARTIFACT_BYTES: u64 = 4_000_000_000;

/// Requires that a ceiling change stayed inside one worker process: every
/// sample across the change shows the original worker alone, and it is still
/// running afterwards. A restart would show a gap or a second worker.
fn assert_worker_process_kept(
    original_worker_process_identifier: u32,
    observed_worker_processes: &[Vec<u32>],
) {
    for observed_identifiers in observed_worker_processes {
        assert_eq!(
            observed_identifiers.as_slice(),
            &[original_worker_process_identifier],
            "the live ceiling change must be adopted by the worker that holds the model; the timeline was {observed_worker_processes:?}"
        );
    }
    assert!(
        is_process_running(original_worker_process_identifier),
        "the worker {original_worker_process_identifier} must keep running across the live ceiling change"
    );
    assert_eq!(
        sole_worker_child_process_identifier(
            "the journey must still own exactly one worker process"
        ),
        original_worker_process_identifier,
        "the same worker process must serve before and after the ceiling change"
    );
}

struct SparseArtifactResidencyGeometry {
    complete_residency_with_headroom_bytes: u64,
    model_geometry: ModelMemoryGeometry,
}

fn measure_sparse_artifact_residency_geometry(
    model_directory: &Path,
) -> SparseArtifactResidencyGeometry {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let (model_geometry, required_headroom_bytes) =
        model_memory_geometry_from_validated_artifact(&validated_artifact, model_directory)
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

fn context_memory_reservation_bytes(model_directory: &Path, context_token_count: usize) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    let reservation_bytes = validated_artifact
        .config()
        .context_memory_reservation_bytes(context_token_count)
        .expect("the acceptance prompt context should fit the model geometry");
    u64::try_from(reservation_bytes)
        .expect("the acceptance prompt context reservation should fit u64")
}

fn artifact_payload_bytes(model_directory: &Path) -> u64 {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, MAXIMUM_OUTPUT_TOKEN_COUNT)
        .expect("the configured sparse artifact should validate");
    validated_artifact.total_payload_bytes()
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
        "ssd_prompt_cache": {"enabled": false},
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
    assert!(
        !assistant_response.trim().is_empty(),
        "{request_label} should return visible model output"
    );
}
