//! Production journey proving a live ceiling below the artifact size still serves.
//!
//! The worker loads the sparse MoE at a ceiling that holds the whole model,
//! then the user lowers the public MLX ceiling far below the artifact payload.
//! The one text engine adopts the lower ceiling in place: the same worker
//! process keeps the model loaded, streams from the SSD what no longer fits,
//! and keeps serving the same conversation. This is the user-facing behavior
//! behind the Settings "Maximum model RAM" control.

use std::path::Path;

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
    configured_installed_model_directory_by_id, exact_model_prompt, resident_sparse_moe_model_id,
};

use super::acceptance_worker_config::write_acceptance_config;
use super::one_token_chat_request::complete_one_token_chat_request;
use super::sparse_artifact_residency::{
    MAXIMUM_REAL_MODEL_MEMORY_BYTES, artifact_payload_bytes, context_memory_reservation_bytes,
    measure_sparse_artifact_residency_geometry, prompt_chunk_mlx_working_buffer_allowance_bytes,
};

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
    let resident_memory_ceiling_bytes = resident_phase_memory_ceiling_bytes(&model_directory);
    let (lowered_ceiling_gb, lowered_ceiling_bytes) =
        lowered_ceiling_below_artifact(artifact_payload_bytes);

    let (worker_home, real_model_rest_server) =
        launch_resident_ceiling_server(model_id, &model_directory, resident_memory_ceiling_bytes)
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

    let initial_floor_bytes = require_resident_status_with_streaming_floor(
        server_address,
        artifact_payload_bytes,
        lowered_ceiling_bytes,
    )
    .await;

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

fn resident_phase_memory_ceiling_bytes(model_directory: &Path) -> u64 {
    let residency_geometry = measure_sparse_artifact_residency_geometry(model_directory);
    // The ceiling purchases complete residency, the resident phases' context,
    // and an allowance for one prompt chunk's MLX working buffers, so every
    // request fits and none streams from the SSD on its own.
    let resident_memory_ceiling_bytes = residency_geometry
        .complete_residency_with_headroom_bytes
        .saturating_add(context_memory_reservation_bytes(
            model_directory,
            RESIDENT_PHASE_CONTEXT_TOKEN_COUNT,
        ))
        .saturating_add(prompt_chunk_mlx_working_buffer_allowance_bytes(
            &residency_geometry.model_geometry,
        ));
    assert!(
        resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
        "the lowered-ceiling acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
    );
    resident_memory_ceiling_bytes
}

/// The lowered ceiling in whole marketed gigabytes and in bytes.
fn lowered_ceiling_below_artifact(artifact_payload_bytes: u64) -> (u64, u64) {
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
    (lowered_ceiling_gb, lowered_ceiling_bytes)
}

/// The isolated worker home the server runs in, and the server itself.
async fn launch_resident_ceiling_server(
    model_id: &str,
    model_directory: &Path,
    resident_memory_ceiling_bytes: u64,
) -> (tempfile::TempDir, RealModelRestServer) {
    let worker_home =
        tempfile::tempdir().expect("the lowered-ceiling worker home should be created");
    write_acceptance_config(
        worker_home.path(),
        model_directory,
        model_id,
        resident_memory_ceiling_bytes,
        false,
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory.to_path_buf(),
        worker_home.path(),
        resident_memory_ceiling_bytes,
    )
    .await;
    (worker_home, real_model_rest_server)
}

/// Returns the geometry floor the loaded model publishes.
async fn require_resident_status_with_streaming_floor(
    server_address: std::net::SocketAddr,
    artifact_payload_bytes: u64,
    lowered_ceiling_bytes: u64,
) -> u64 {
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
    initial_floor_bytes
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

async fn serve_and_require_output(
    server_address: std::net::SocketAddr,
    openai_client: &LocalOpenAiClient,
    model_id: &str,
    messages: Value,
    request_label: &str,
) {
    complete_one_token_chat_request(
        server_address,
        openai_client,
        model_id,
        messages,
        request_label,
    )
    .await;
}
