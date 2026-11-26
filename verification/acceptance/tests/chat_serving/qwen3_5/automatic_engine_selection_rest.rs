//! Production-worker journeys for artifact-derived Qwen3.5 engine selection.

use std::{fs, path::Path, time::Duration};

use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, get_json_endpoint, launch_real_model_rest_server,
    stop_real_model_rest_server,
};
use crate::support::{
    configured_installed_model_directory_by_id, exact_model_prompt,
    large_sparse_moe_streaming_model_id, resident_sparse_moe_model_id,
};

use super::acceptance_worker_config::write_acceptance_config;
use super::one_token_chat_request::complete_one_token_chat_request;
pub(super) use super::sparse_artifact_residency::MAXIMUM_REAL_MODEL_MEMORY_BYTES;
use super::sparse_artifact_residency::{
    context_memory_reservation_bytes, measure_sparse_artifact_residency_geometry,
    prompt_chunk_mlx_working_buffer_allowance_bytes,
};

const SMALL_PROMPT_TOKEN_COUNT: usize = 256;
const SHORT_PROMPT_TOKEN_COUNT: usize = 40;
const MULTI_CHUNK_PROMPT_TOKEN_COUNT: usize = 5_000;
/// Shared with the speculative-decoding sibling journey in this module.
pub(super) const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");
const WHOLE_MODEL_JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test(flavor = "multi_thread")]
#[ignore = "starts the production worker and loads the configured sparse artifact whole"]
#[serial]
async fn should_hold_the_whole_model_when_artifact_geometry_and_prompt_fit() {
    timeout(WHOLE_MODEL_JOURNEY_TIMEOUT, async {
        let model_id = resident_sparse_moe_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Explain how haste shapes the tragedy in one concise sentence.",
            SMALL_PROMPT_TOKEN_COUNT,
        );
        let prompt_context_bytes =
            context_memory_reservation_bytes(&model_directory, SMALL_PROMPT_TOKEN_COUNT + 2);
        // The ceiling has to hold one prompt chunk's MLX working buffers on top
        // of the startup residency terms, or the first request sheds experts
        // to make room for them (issue #1508).
        let prefill_activation_workspace_bytes =
            prompt_chunk_mlx_working_buffer_allowance_bytes(&residency_geometry.model_geometry);
        let resident_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_add(prompt_context_bytes)
            .saturating_add(prefill_activation_workspace_bytes);
        assert!(
            resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
            "the resident acceptance cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
        );

        let worker_home = tempfile::tempdir()
            .expect("the whole-model worker home should be created");
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            resident_memory_ceiling_bytes,
            false,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            resident_memory_ceiling_bytes,
        )
        .await;
        complete_chat_request(
            real_model_rest_server.server_address,
            model_id,
            json!([{"role": "user", "content": prompt_content}]),
            "resident-selection",
        )
        .await;
        let status_document =
            get_json_endpoint(real_model_rest_server.server_address, "/v1/status").await;
        assert_eq!(status_document["expert_memory_mode"], "resident");
        assert_eq!(status_document["activity"], "idle");
        let resident_expert_payload_bytes = status_document["expert_residency"]
            ["resident_expert_payload_bytes"]
            .as_u64()
            .expect("resident status should report its owned expert payload");
        assert!(resident_expert_payload_bytes > 0);
        stop_real_model_rest_server(real_model_rest_server).await;
    })
    .await
    .expect("the whole-model journey must finish within 115 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "loads the whole model, then sends a short and a multi-chunk prompt that both fit"]
#[serial]
async fn should_stay_resident_for_a_multi_chunk_prompt_after_a_short_prompt() {
    timeout(SSD_JOURNEY_TIMEOUT, async {
        let model_id = resident_sparse_moe_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let short_user_message = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Name the two families in one short sentence.",
            SHORT_PROMPT_TOKEN_COUNT,
        );
        let multi_chunk_user_message = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Explain how haste shapes the tragedy in one concise sentence.",
            MULTI_CHUNK_PROMPT_TOKEN_COUNT,
        );
        // The ceiling purchases complete residency, the multi-chunk context,
        // and an allowance for one prompt chunk's MLX working buffers, so the
        // second request fits.
        let resident_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_add(context_memory_reservation_bytes(
                &model_directory,
                MULTI_CHUNK_PROMPT_TOKEN_COUNT + 2,
            ))
            .saturating_add(prompt_chunk_mlx_working_buffer_allowance_bytes(
                &residency_geometry.model_geometry,
            ));
        assert!(
            resident_memory_ceiling_bytes <= MAXIMUM_REAL_MODEL_MEMORY_BYTES,
            "the multi-chunk resident cell must remain within 36 GB: ceiling={resident_memory_ceiling_bytes}"
        );
        let worker_home = tempfile::tempdir()
            .expect("the multi-chunk resident worker home should be created");
        // The `astro_ssd_prompt_cache` stops a terminal chunk one token short,
        // so the short request leaves a one-token prefill observation. That
        // observation once extrapolated to a 2,048-token chunk as a ~23 GB
        // phantom transient and moved a fitting request to SSD streaming.
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            resident_memory_ceiling_bytes,
            true,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            resident_memory_ceiling_bytes,
        )
        .await;
        let server_address = real_model_rest_server.server_address;
        complete_chat_request(
            server_address,
            model_id,
            json!([{"role": "user", "content": short_user_message}]),
            "short-resident-prompt",
        )
        .await;
        complete_chat_request(
            server_address,
            model_id,
            json!([{"role": "user", "content": multi_chunk_user_message}]),
            "multi-chunk-resident-prompt",
        )
        .await;
        let status_document = get_json_endpoint(server_address, "/v1/status").await;
        assert_eq!(
            status_document["expert_memory_mode"], "resident",
            "a fitting multi-chunk prompt must leave every expert in memory: {status_document}"
        );
        stop_real_model_rest_server(real_model_rest_server).await;
    })
    .await
    .expect("the multi-chunk resident journey must finish within 120 seconds");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "starts the production worker with an artifact-derived ceiling below complete-residency headroom"]
#[serial]
async fn should_select_streaming_engine_when_complete_residency_headroom_does_not_fit() {
    timeout(SSD_JOURNEY_TIMEOUT, async {
        let model_id = large_sparse_moe_streaming_model_id();
        let model_directory = configured_installed_model_directory_by_id(model_id);
        let residency_geometry =
            measure_sparse_artifact_residency_geometry(&model_directory);
        let streaming_memory_ceiling_bytes = residency_geometry
            .complete_residency_with_headroom_bytes
            .saturating_sub(1)
            .min(MAXIMUM_REAL_MODEL_MEMORY_BYTES);
        let prompt_content = exact_model_prompt::build_exact_model_prompt_content(
            &model_directory,
            ROMEO_AND_JULIET_SOURCE,
            "Explain how haste shapes the tragedy in one concise sentence.",
            SMALL_PROMPT_TOKEN_COUNT,
        );
        let worker_home = tempfile::tempdir()
            .expect("the engine-selection worker home should be created");
        write_acceptance_config(
            worker_home.path(),
            &model_directory,
            model_id,
            streaming_memory_ceiling_bytes,
            false,
        );
        let real_model_rest_server = launch_real_model_rest_server(
            model_id,
            model_directory,
            worker_home.path(),
            streaming_memory_ceiling_bytes,
        )
        .await;
        complete_chat_request(
            real_model_rest_server.server_address,
            model_id,
            json!([{"role": "user", "content": prompt_content}]),
            "streaming-selection",
        )
        .await;
        let status_document =
            get_json_endpoint(real_model_rest_server.server_address, "/v1/status").await;
        assert!(
            matches!(
                status_document["expert_memory_mode"].as_str(),
                Some("paged" | "hybrid")
            ),
            "the engine selected below complete-residency headroom must remain nonresident: {status_document}"
        );
        assert!(
            status_document["expert_residency"].get("complete_layer_count").is_none()
                && status_document["expert_residency"]
                    .get("partial_layer_count")
                    .is_none(),
            "streaming status must not publish complete-layer ownership: {status_document}"
        );
        stop_real_model_rest_server(real_model_rest_server).await;
    })
    .await
    .expect("the engine-selection SSD journey must finish within 120 seconds");
}

async fn complete_chat_request(
    server_address: std::net::SocketAddr,
    model_id: &str,
    messages: Value,
    request_label: &str,
) -> String {
    let openai_client = LocalOpenAiClient::new(server_address, "local-acceptance-client");
    complete_one_token_chat_request(
        server_address,
        &openai_client,
        model_id,
        messages,
        request_label,
    )
    .await
}

pub(super) fn read_generation_attribution_reports(worker_home: &Path) -> Vec<Value> {
    let attribution_log_path =
        worker_home.join(".astronomical-dev/logs/performance-attribution.jsonl");
    fs::read_to_string(&attribution_log_path)
        .unwrap_or_else(|read_error| {
            panic!(
                "{} should be readable after the worker stops: {read_error}",
                attribution_log_path.display()
            )
        })
        .lines()
        .map(|json_line| {
            serde_json::from_str::<Value>(json_line)
                .expect("each acceptance attribution row should be valid JSON")
        })
        .filter(|report| report["report_kind"] == "generation")
        .collect()
}
