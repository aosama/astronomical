//! One conversation while the limit is raised and lowered again.
//!
//! A person starts under a limit that makes the model stream, raises the
//! limit until every expert fits, then lowers it again. Each turn must be
//! served, the status page must never contradict itself, nearly every expert
//! must stay in memory after the raise, and the model must stream again
//! after the limit comes back down.
//!
//! The limit changes while one production worker is loaded and idle. Each
//! follow-up resends the exact ordered user and assistant history, matching
//! the stateless OpenAI conversation contract rather than relying on
//! test-local engine state.

mod support;

use std::{net::SocketAddr, time::Instant};

use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::{Duration, interval, timeout};

use crate::support::openai_client::{ChatCompletionStream, LocalOpenAiClient};
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, launch_real_model_rest_server, put_json_endpoint,
    stop_real_model_rest_server,
};
use support::{
    BYTES_PER_MARKETED_GIGABYTE, assert_machine_supports_round_trip,
    assert_most_experts_stay_in_memory, assert_nonresident_status_before_request,
    assert_streaming_status, kept_expert_payload_bytes, read_generation_evidence,
    wait_for_idle_status, wait_for_settled_resident_status, write_round_trip_config,
};

pub(super) const LOG_MARKER: &str = "[live-memory-ceiling-residency-round-trip]";

const SETTLE_TIMEOUT: Duration = Duration::from_secs(45);

fn model_id() -> &'static str {
    crate::support::large_sparse_moe_streaming_model_id()
}
const INITIAL_MLX_MEMORY_CEILING_BYTES: u64 = 23_000_000_000;
// The REST raise sends whole GB; the daemon converts user GB input with the
// marketed-gigabyte convention, so the expected effective ceiling is 38 x 2^30.
const RESIDENT_MLX_MEMORY_CEILING_BYTES: u64 = 38 * BYTES_PER_MARKETED_GIGABYTE;
// Below the large sparse MoE streaming complete-owner idle snapshot (~28.5 GB). 30 GB used to look
// too small only because admission stacked exclusive paper peaks.
const RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES: u64 = 26 * BYTES_PER_MARKETED_GIGABYTE;
const INITIAL_PROMPT_TOKEN_COUNT: usize = 1_024;
pub(super) const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 16;
const THINKING_BUDGET_TOKEN_COUNT: u32 = 0;
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one production worker and changes its public MLX ceiling twice"]
#[serial]
async fn should_serve_one_conversation_across_streaming_resident_and_streaming_memory_limits() {
    timeout(SSD_JOURNEY_TIMEOUT, run_residency_round_trip())
        .await
        .expect("the live-ceiling expert-residency SSD journey must finish within 120 seconds");
}

async fn run_residency_round_trip() {
    let journey_started_at = Instant::now();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id());
    let isolated_worker_home =
        tempfile::tempdir().expect("the live-ceiling round-trip worker home should be created");
    write_round_trip_config(
        isolated_worker_home.path(),
        &model_directory,
        INITIAL_MLX_MEMORY_CEILING_BYTES,
    );
    let initial_user_message = crate::support::exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        ROMEO_AND_JULIET_SOURCE,
        "Explain how haste shapes the tragedy in one concise paragraph.",
        INITIAL_PROMPT_TOKEN_COUNT,
    );
    eprintln!(
        "{LOG_MARKER} phase=server_start status=start initial_ceiling_gb=23 resident_ceiling_gb=38 final_ceiling_gb=26 timeout_seconds={}",
        SSD_JOURNEY_TIMEOUT.as_secs()
    );
    let real_model_rest_server = launch_real_model_rest_server(
        model_id(),
        model_directory,
        isolated_worker_home.path(),
        INITIAL_MLX_MEMORY_CEILING_BYTES,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let initial_status = wait_for_idle_status(server_address, "initial_idle").await;
    assert_machine_supports_round_trip(&initial_status, RESIDENT_MLX_MEMORY_CEILING_BYTES);
    let openai_client =
        LocalOpenAiClient::new(server_address, "local-live-memory-round-trip-client");

    let first_assistant_response = execute_conversation_request(
        &openai_client,
        json!([{"role": "user", "content": initial_user_message}]),
        "turn_1_streaming",
    )
    .await;
    let first_status = wait_for_idle_status(server_address, "turn_1_finalization").await;
    crate::support::memory_utilization_parity::assert_and_preserve_status_memory_ceiling_utilization(
        "live-memory-ceiling-round-trip",
        isolated_worker_home.path(),
        &first_status,
    );
    assert_streaming_status(&first_status, INITIAL_MLX_MEMORY_CEILING_BYTES);
    let first_evidence = read_generation_evidence(
        isolated_worker_home.path(),
        &[INITIAL_MLX_MEMORY_CEILING_BYTES],
    );
    assert!(first_evidence[0].expert_source_read_bytes > 0);

    let conversation = RoundTripConversation {
        server_address,
        openai_client: &openai_client,
        worker_home: isolated_worker_home.path(),
        initial_user_message: &initial_user_message,
        first_assistant_response: &first_assistant_response,
    };
    let (resident_ceiling_bytes, second_assistant_response) =
        raise_the_limit_and_serve_the_second_turn(&conversation).await;
    let return_to_streaming_ceiling_bytes =
        lower_the_limit_and_serve_the_third_turn(&conversation, &second_assistant_response).await;
    stop_real_model_rest_server(real_model_rest_server).await;

    let generation_evidence = read_generation_evidence(
        isolated_worker_home.path(),
        &[
            INITIAL_MLX_MEMORY_CEILING_BYTES,
            resident_ceiling_bytes,
            return_to_streaming_ceiling_bytes,
        ],
    );
    assert!(generation_evidence[2].expert_source_read_bytes > 0);
    assert_generation_continuity(&generation_evidence[1], &generation_evidence[2]);
    eprintln!(
        "{LOG_MARKER} phase=journey status=success elapsed_seconds={:.3} turn_1_expert_read_gb={:.3} turn_2_expert_read_gb={:.3} turn_3_expert_read_gb={:.3}",
        journey_started_at.elapsed().as_secs_f64(),
        generation_evidence[0].expert_source_read_bytes as f64 / 1_073_741_824.0,
        generation_evidence[1].expert_source_read_bytes as f64 / 1_073_741_824.0,
        generation_evidence[2].expert_source_read_bytes as f64 / 1_073_741_824.0,
    );
}

/// What the later turns of the conversation need from the first.
struct RoundTripConversation<'journey> {
    server_address: SocketAddr,
    openai_client: &'journey LocalOpenAiClient,
    worker_home: &'journey std::path::Path,
    initial_user_message: &'journey str,
    first_assistant_response: &'journey str,
}

const SECOND_USER_MESSAGE: &str =
    "Which decision best demonstrates that haste, and why? Answer briefly.";

/// Raises the limit until every expert fits while idle, then serves the
/// second turn. Returns the limit that was applied and the reply.
async fn raise_the_limit_and_serve_the_second_turn(
    conversation: &RoundTripConversation<'_>,
) -> (u64, String) {
    let server_address = conversation.server_address;
    let resident_ceiling_bytes =
        apply_memory_ceiling(server_address, 38, RESIDENT_MLX_MEMORY_CEILING_BYTES).await;
    // Promotion replaces the worker so the in-memory engine loads under the
    // raised limit. The settle wait tolerates the replacement's loading
    // window and asserts the resident contract against the applied limit.
    let resident_status = wait_for_settled_resident_status(
        server_address,
        "resident_before_turn_2",
        resident_ceiling_bytes,
        SETTLE_TIMEOUT,
    )
    .await;
    let complete_expert_payload_bytes = kept_expert_payload_bytes(&resident_status);
    let second_assistant_response = execute_conversation_request(
        conversation.openai_client,
        json!([
            {"role": "user", "content": conversation.initial_user_message},
            {"role": "assistant", "content": conversation.first_assistant_response},
            {"role": "user", "content": SECOND_USER_MESSAGE}
        ]),
        "turn_2_resident",
    )
    .await;
    assert!(!second_assistant_response.is_empty());
    let second_status = wait_for_idle_status(server_address, "turn_2_finalization").await;
    // The raised limit holds every expert while idle. Whether it also holds
    // a turn's stored state and MLX working buffers depend on the limit the
    // machine grants, so the engine may give a few experts back to serve
    // the turn. What a person is owed is that the turn is served and that
    // nearly everything stays in memory.
    assert_most_experts_stay_in_memory(
        &second_status,
        resident_ceiling_bytes,
        complete_expert_payload_bytes,
    );
    let second_evidence = read_generation_evidence(
        conversation.worker_home,
        &[INITIAL_MLX_MEMORY_CEILING_BYTES, resident_ceiling_bytes],
    );
    assert_generation_continuity(&second_evidence[0], &second_evidence[1]);
    (resident_ceiling_bytes, second_assistant_response)
}

/// Lowers the limit below what every expert needs and serves the third
/// turn, which must stream again. Returns the limit that was applied.
async fn lower_the_limit_and_serve_the_third_turn(
    conversation: &RoundTripConversation<'_>,
    second_assistant_response: &str,
) -> u64 {
    let server_address = conversation.server_address;
    let return_to_streaming_ceiling_bytes = apply_memory_ceiling(
        server_address,
        26,
        RETURN_TO_STREAMING_MLX_MEMORY_CEILING_BYTES,
    )
    .await;
    let paged_status = wait_for_idle_status(server_address, "streaming_before_turn_3").await;
    assert_nonresident_status_before_request(&paged_status, return_to_streaming_ceiling_bytes);
    let third_assistant_response = execute_conversation_request(
        conversation.openai_client,
        json!([
            {"role": "user", "content": conversation.initial_user_message},
            {"role": "assistant", "content": conversation.first_assistant_response},
            {"role": "user", "content": SECOND_USER_MESSAGE},
            {"role": "assistant", "content": second_assistant_response},
            {"role": "user", "content": "Relate that decision to the final reconciliation. Keep the answer concise."}
        ]),
        "turn_3_streaming",
    )
    .await;
    assert!(!third_assistant_response.is_empty());
    let third_status = wait_for_idle_status(server_address, "turn_3_finalization").await;
    assert_streaming_status(&third_status, return_to_streaming_ceiling_bytes);
    return_to_streaming_ceiling_bytes
}

async fn execute_conversation_request(
    openai_client: &LocalOpenAiClient,
    conversation_messages: Value,
    request_label: &str,
) -> String {
    let request_started_at = Instant::now();
    eprintln!("{LOG_MARKER} phase={request_label} status=start");
    let request_document = json!({
        "model": model_id(),
        "messages": conversation_messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "thinking_budget": THINKING_BUDGET_TOKEN_COUNT,
    });
    let mut stream: ChatCompletionStream = openai_client
        .create_streaming_chat_completion(&request_document)
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

/// Applies a ceiling through the public endpoint and returns the effective
/// ceiling the worker adopted.
///
/// A raise that promotes the worker to the resident engine restarts it, and
/// the resident engine reports its claimed wired-residency budget as the
/// effective ceiling when that budget lands below the configured value, so
/// callers must assert subsequent status against the applied value this
/// returns rather than against the configured GB conversion.
async fn apply_memory_ceiling(
    server_address: SocketAddr,
    maximum_mlx_memory_gb: u64,
    configured_ceiling_bytes: u64,
) -> u64 {
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
    let applied_ceiling_bytes = response_document["effective_mlx_memory_ceiling_bytes"]
        .as_u64()
        .unwrap_or(0);
    assert_eq!(
        response_document["configured_maximum_mlx_memory_gb"],
        maximum_mlx_memory_gb
    );
    assert!(
        applied_ceiling_bytes > 0 && applied_ceiling_bytes <= configured_ceiling_bytes,
        "applied ceiling {applied_ceiling_bytes} should be positive and within the configured {configured_ceiling_bytes}"
    );
    assert!(response_document["pending_mlx_memory_ceiling_bytes"].is_null());
    eprintln!(
        "{LOG_MARKER} phase=ceiling_{maximum_mlx_memory_gb}_gb status=success elapsed_seconds={:.3} applied_ceiling_bytes={applied_ceiling_bytes}",
        transition_started_at.elapsed().as_secs_f64(),
    );
    applied_ceiling_bytes
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
