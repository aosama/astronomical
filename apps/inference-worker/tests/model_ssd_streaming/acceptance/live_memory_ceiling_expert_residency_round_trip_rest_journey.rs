//! Public journey for one conversation across SSD streaming, complete residency, and renewed streaming.
//!
//! The ceiling transitions occur while one production worker remains loaded and
//! idle. Each follow-up resends the exact ordered user/assistant history, matching
//! the stateless OpenAI conversation contract rather than relying on test-local
//! engine state.

mod support;

use std::{net::SocketAddr, time::Instant};

use async_openai::{Client, config::OpenAIConfig, types::stream::StreamResponse};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::time::{Duration, interval, timeout};

use crate::common::real_model_rest_server::{
    JOURNEY_TIMEOUT, launch_real_model_rest_server, put_json_endpoint, stop_real_model_rest_server,
};
use support::{
    assert_machine_supports_round_trip, assert_nonresident_status_before_request,
    assert_resident_status, assert_streaming_status, read_generation_evidence,
    wait_for_idle_status, write_round_trip_config,
};

pub(super) const LOG_MARKER: &str = "[live-memory-ceiling-residency-round-trip]";
const MODEL_ID: &str = crate::common::ORNITH_SSD_STREAMING_MODEL_ID;
const INITIAL_MLX_MEMORY_CEILING_BYTES: u64 = 23_000_000_000;
const RESIDENT_MLX_MEMORY_CEILING_BYTES: u64 = 38_000_000_000;
// Below the oQ6e complete-owner idle snapshot (~28.5 GB). 30 GB used to look
// too small only because admission stacked exclusive paper peaks.
const RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES: u64 = 26_000_000_000;
const INITIAL_PROMPT_TOKEN_COUNT: usize = 1_024;
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 16;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 0;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one production worker and changes its public MLX ceiling twice"]
async fn should_serve_one_conversation_across_streaming_resident_and_streaming_memory_limits() {
    timeout(JOURNEY_TIMEOUT, run_residency_round_trip())
        .await
        .expect("the live-ceiling expert-residency round trip must finish within 115 seconds");
}

async fn run_residency_round_trip() {
    let journey_started_at = Instant::now();
    let model_directory = crate::common::configured_model_artifact_directory_by_id(MODEL_ID);
    let isolated_worker_home =
        tempfile::tempdir().expect("the live-ceiling round-trip worker home should be created");
    write_round_trip_config(
        isolated_worker_home.path(),
        &model_directory,
        INITIAL_MLX_MEMORY_CEILING_BYTES,
    );
    let initial_user_message = crate::common::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Explain how haste shapes the tragedy in one concise paragraph.",
        INITIAL_PROMPT_TOKEN_COUNT,
    );
    eprintln!(
        "{LOG_MARKER} phase=server_start status=start initial_ceiling_gb=23 resident_ceiling_gb=38 final_ceiling_gb=26 timeout_seconds={}",
        JOURNEY_TIMEOUT.as_secs()
    );
    let real_model_rest_server = launch_real_model_rest_server(
        MODEL_ID,
        model_directory,
        isolated_worker_home.path(),
        INITIAL_MLX_MEMORY_CEILING_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let initial_status = wait_for_idle_status(server_address, "initial_idle").await;
    assert_machine_supports_round_trip(&initial_status, RESIDENT_MLX_MEMORY_CEILING_BYTES);
    let openai_client = Client::with_config(
        OpenAIConfig::new()
            .with_api_base(format!("http://{server_address}/v1"))
            .with_api_key("local-live-memory-round-trip-client"),
    );

    let first_assistant_response = execute_conversation_request(
        &openai_client,
        json!([{"role": "user", "content": initial_user_message}]),
        "turn_1_streaming",
    )
    .await;
    let first_status = wait_for_idle_status(server_address, "turn_1_finalization").await;
    assert_streaming_status(&first_status, INITIAL_MLX_MEMORY_CEILING_BYTES);
    let first_evidence = read_generation_evidence(
        isolated_worker_home.path(),
        &[INITIAL_MLX_MEMORY_CEILING_BYTES],
    );
    assert!(first_evidence[0].expert_source_read_bytes > 0);

    apply_memory_ceiling(server_address, 38, RESIDENT_MLX_MEMORY_CEILING_BYTES).await;
    let resident_status = wait_for_idle_status(server_address, "resident_before_turn_2").await;
    assert_resident_status(&resident_status, RESIDENT_MLX_MEMORY_CEILING_BYTES);

    let second_user_message =
        "Which decision best demonstrates that haste, and why? Answer briefly.";
    let second_assistant_response = execute_conversation_request(
        &openai_client,
        json!([
            {"role": "user", "content": initial_user_message},
            {"role": "assistant", "content": first_assistant_response},
            {"role": "user", "content": second_user_message}
        ]),
        "turn_2_resident",
    )
    .await;
    let second_status = wait_for_idle_status(server_address, "turn_2_finalization").await;
    assert_resident_status(&second_status, RESIDENT_MLX_MEMORY_CEILING_BYTES);
    let second_evidence = read_generation_evidence(
        isolated_worker_home.path(),
        &[
            INITIAL_MLX_MEMORY_CEILING_BYTES,
            RESIDENT_MLX_MEMORY_CEILING_BYTES,
        ],
    );
    assert_eq!(second_evidence[1].expert_source_read_bytes, 0);
    assert_generation_continuity(&second_evidence[0], &second_evidence[1]);

    apply_memory_ceiling(
        server_address,
        26,
        RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES,
    )
    .await;
    let paged_status = wait_for_idle_status(server_address, "streaming_before_turn_3").await;
    assert_nonresident_status_before_request(
        &paged_status,
        RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES,
    );

    let third_user_message =
        "Relate that decision to the final reconciliation. Keep the answer concise.";
    let third_assistant_response = execute_conversation_request(
        &openai_client,
        json!([
            {"role": "user", "content": initial_user_message},
            {"role": "assistant", "content": first_assistant_response},
            {"role": "user", "content": second_user_message},
            {"role": "assistant", "content": second_assistant_response},
            {"role": "user", "content": third_user_message}
        ]),
        "turn_3_streaming",
    )
    .await;
    assert!(!third_assistant_response.is_empty());
    let third_status = wait_for_idle_status(server_address, "turn_3_finalization").await;
    assert_streaming_status(&third_status, RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES);
    stop_real_model_rest_server(real_model_rest_server).await;

    let generation_evidence = read_generation_evidence(
        isolated_worker_home.path(),
        &[
            INITIAL_MLX_MEMORY_CEILING_BYTES,
            RESIDENT_MLX_MEMORY_CEILING_BYTES,
            RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES,
        ],
    );
    assert!(generation_evidence[2].expert_source_read_bytes > 0);
    assert_generation_continuity(&generation_evidence[1], &generation_evidence[2]);
    eprintln!(
        "{LOG_MARKER} phase=journey status=success elapsed_seconds={:.3} turn_1_expert_read_gb={:.3} turn_2_expert_read_gb={:.3} turn_3_expert_read_gb={:.3}",
        journey_started_at.elapsed().as_secs_f64(),
        generation_evidence[0].expert_source_read_bytes as f64 / 1_000_000_000.0,
        generation_evidence[1].expert_source_read_bytes as f64 / 1_000_000_000.0,
        generation_evidence[2].expert_source_read_bytes as f64 / 1_000_000_000.0,
    );
}

async fn execute_conversation_request(
    openai_client: &Client<OpenAIConfig>,
    conversation_messages: Value,
    request_label: &str,
) -> String {
    let request_started_at = Instant::now();
    eprintln!("{LOG_MARKER} phase={request_label} status=start");
    let request_document = json!({
        "model": MODEL_ID,
        "messages": conversation_messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    let mut stream: StreamResponse<Value> = openai_client
        .chat()
        .create_stream_byot(request_document)
        .await
        .unwrap_or_else(|request_error| panic!("{request_label} should start: {request_error}"));
    let mut assistant_text = String::new();
    let mut completion_finish_reason = None;
    while let Some(stream_item) = stream.next().await {
        let stream_chunk = stream_item
            .unwrap_or_else(|stream_error| panic!("{request_label} should stream: {stream_error}"));
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                assistant_text.push_str(content_fragment);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                completion_finish_reason = Some(reason.to_owned());
            }
        }
    }
    let assistant_text = assistant_text.trim().to_owned();
    assert!(
        !assistant_text.is_empty(),
        "{request_label} should produce visible text"
    );
    assert!(matches!(
        completion_finish_reason.as_deref(),
        Some("stop" | "length")
    ));
    eprintln!(
        "{LOG_MARKER} phase={request_label} status=success elapsed_seconds={:.3} output_characters={}",
        request_started_at.elapsed().as_secs_f64(),
        assistant_text.len(),
    );
    assistant_text
}

async fn apply_memory_ceiling(
    server_address: SocketAddr,
    maximum_mlx_memory_gb: u64,
    expected_ceiling_bytes: u64,
) {
    let transition_started_at = Instant::now();
    eprintln!("{LOG_MARKER} phase=ceiling_{maximum_mlx_memory_gb}_gb status=start");
    let request_body = json!({"maximum_mlx_memory_gb": maximum_mlx_memory_gb});
    let transition_future = put_json_endpoint(
        server_address,
        "/v1/config/maximum-mlx-memory",
        &request_body,
    );
    tokio::pin!(transition_future);
    let mut progress_interval = interval(Duration::from_secs(1));
    progress_interval.tick().await;
    let response_document = loop {
        tokio::select! {
            response_document = &mut transition_future => break response_document,
            _ = progress_interval.tick() => {
                eprintln!(
                    "{LOG_MARKER} phase=ceiling_{maximum_mlx_memory_gb}_gb status=progress elapsed_seconds={:.3}",
                    transition_started_at.elapsed().as_secs_f64(),
                );
            }
        }
    };
    assert_eq!(
        response_document["effective_mlx_memory_ceiling_bytes"],
        expected_ceiling_bytes
    );
    assert_eq!(
        response_document["configured_maximum_mlx_memory_gb"],
        maximum_mlx_memory_gb
    );
    assert!(response_document["pending_mlx_memory_ceiling_bytes"].is_null());
    eprintln!(
        "{LOG_MARKER} phase=ceiling_{maximum_mlx_memory_gb}_gb status=success elapsed_seconds={:.3}",
        transition_started_at.elapsed().as_secs_f64(),
    );
}

fn assert_generation_continuity(
    previous_generation: &support::GenerationEvidence,
    current_generation: &support::GenerationEvidence,
) {
    assert_eq!(current_generation.model_id, previous_generation.model_id);
    assert_eq!(
        current_generation.model_revision,
        previous_generation.model_revision
    );
    assert!(
        current_generation.prompt_token_count > previous_generation.prompt_token_count,
        "resending ordered conversation history must increase prompt context"
    );
}
