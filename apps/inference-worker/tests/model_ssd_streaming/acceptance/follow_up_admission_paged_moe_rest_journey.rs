//! REST acceptance: the follow-up-turn admission journey on the paged large sparse MoE.
//!
//! Production failed exactly here (2026-10-10): a long first turn taught the
//! adaptive RAM budget chunk-shaped activation observations, and the follow-up
//! turn in the same conversation was rejected five times with `generation
//! context exceeds available GPU wired memory` because admission inflated a
//! smaller-scope observation by the chunk ratio and capped the reserve at the
//! whole ceiling. This journey replays that user journey against the same
//! artifact class under a paged ceiling and requires both turns to complete.

use std::fs;
use std::path::Path;

use futures_util::StreamExt;
use serde_json::{Value, json};
use serial_test::serial;
use tokio::time::timeout;

use crate::support::artifact_bytes::{
    acceptance_model_scan_root, artifact_directory_regular_file_bytes,
};
use crate::support::exact_model_prompt;
use crate::support::openai_client::LocalOpenAiClient;
use crate::support::serving_rest::{
    SSD_JOURNEY_TIMEOUT, launch_real_model_rest_server, stop_real_model_rest_server,
};
use crate::support::{acceptance_evidence, configured_installed_model_directory_by_id};

const LOG_MARKER: &str = "[follow-up-admission-paged-moe]";
const MAXIMUM_OUTPUT_TOKEN_COUNT: u32 = 128;
/// The regression guard shape: the resident chunk teaches the budget at its
/// own operation scope, while a wider-than-resident SSD-streaming chunk is
/// the bound the paged follow-up admission resolves at. With the pre-fix code
/// this pair inflated the reserve by the chunk ratio; with the fix each mode
/// resolves its own scope, so both turns must simply complete.
const RESIDENT_CHUNK_TOKENS: u32 = 2_048;
const SSD_STREAMING_CHUNK_TOKENS: u32 = 4_096;
const MAXIMUM_MODEL_TEST_MEMORY_BYTES: u64 = 36_000_000_000;
const TEN_THOUSAND_TOKEN_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");
const TURN_ONE_INSTRUCTION: &str =
    "Summarize the play's central conflict and the decisions that produce its tragic ending.";
const FOLLOW_UP_ASSISTANT_BRIDGE: &str = "Understood, I have the excerpt in mind.";
const FOLLOW_UP_QUESTION: &str =
    "In one sentence: who wrote the excerpt and what is its central conflict?";
const REJECTION_SUBSTRING: &str = "generation context exceeds available GPU wired memory";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches one production worker against the large sparse MoE paged fixture and requires both conversation turns to complete"]
#[serial]
async fn should_admit_the_paged_moe_follow_up_turn_after_a_long_prefill() {
    timeout(SSD_JOURNEY_TIMEOUT, run_follow_up_admission_journey())
        .await
        .expect("the paged MoE follow-up admission journey must finish within 60 seconds");
}

async fn run_follow_up_admission_journey() {
    let model_id = crate::support::large_sparse_moe_model_id();
    let model_directory = configured_installed_model_directory_by_id(model_id);
    // Persistent (not tempfile) so worker logs survive a timeout or panic.
    let isolated_worker_home_path =
        acceptance_evidence::acceptance_evidence_root("follow-up-admission-paged-moe")
            .join("worker-home");
    fs::remove_dir_all(&isolated_worker_home_path).ok();
    fs::create_dir_all(&isolated_worker_home_path)
        .expect("the persistent worker home should be created");
    let isolated_worker_home_path = isolated_worker_home_path
        .canonicalize()
        .unwrap_or(isolated_worker_home_path);
    let artifact_payload_bytes = artifact_directory_regular_file_bytes(&model_directory);
    // A ceiling below the validated artifact payload forces the paged regime.
    let ceiling_bytes =
        (artifact_payload_bytes.saturating_mul(87) / 100).min(MAXIMUM_MODEL_TEST_MEMORY_BYTES);
    let ceiling_gb = ceiling_bytes / 1_000_000_000;
    let ceiling_bytes = ceiling_gb * 1_000_000_000;
    assert!(
        ceiling_bytes > 0
            && ceiling_bytes < artifact_payload_bytes
            && ceiling_bytes <= MAXIMUM_MODEL_TEST_MEMORY_BYTES,
        "the discovered artifact must admit a paged ceiling within the model-test budget"
    );
    write_follow_up_admission_config(&isolated_worker_home_path, &model_directory, ceiling_gb);
    eprintln!(
        "{LOG_MARKER} request=journey status=start timeout_seconds={} artifact_payload_gb={:.3} ceiling_gb={ceiling_gb} resident_chunk_tokens={RESIDENT_CHUNK_TOKENS} ssd_streaming_chunk_tokens={SSD_STREAMING_CHUNK_TOKENS} persistent_prompt_cache_enabled=true",
        SSD_JOURNEY_TIMEOUT.as_secs(),
        artifact_payload_bytes as f64 / 1_000_000_000.0,
    );

    let real_model_rest_server = launch_real_model_rest_server(
        model_id,
        model_directory.clone(),
        &isolated_worker_home_path,
        ceiling_bytes,
    )
    .await;
    let server_address = real_model_rest_server.server_address;
    let openai_client = LocalOpenAiClient::new(server_address, "follow-up-admission-client");

    let first_turn_prompt = exact_model_prompt::build_exact_model_prompt_content(
        &model_directory,
        TEN_THOUSAND_TOKEN_ROMEO_AND_JULIET_SOURCE,
        TURN_ONE_INSTRUCTION,
        10_000,
    );
    let first_turn_outcome = execute_follow_up_journey_request(
        &openai_client,
        "first_turn",
        json!({
            "model": model_id,
            "messages": [{"role": "user", "content": first_turn_prompt}],
            "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
            "temperature": 1,
            "thinking_budget": 0,
            "stream": true,
        }),
    )
    .await;

    let follow_up_outcome = execute_follow_up_journey_request(
        &openai_client,
        "follow_up",
        json!({
            "model": model_id,
            "messages": [
                {"role": "user", "content": first_turn_prompt},
                {"role": "assistant", "content": FOLLOW_UP_ASSISTANT_BRIDGE},
                {"role": "user", "content": FOLLOW_UP_QUESTION},
            ],
            "max_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
            "temperature": 1,
            "thinking_budget": 0,
            "stream": true,
        }),
    )
    .await;
    stop_real_model_rest_server(real_model_rest_server).await;

    assert_completed_follow_up_journey_request("first_turn", &first_turn_outcome);
    assert_completed_follow_up_journey_request("follow_up", &follow_up_outcome);
    eprintln!(
        "{LOG_MARKER} request=journey status=success first_turn_tokens={} follow_up_tokens={} first_turn_fragments={} follow_up_fragments={}",
        first_turn_outcome.generated_token_count,
        follow_up_outcome.generated_token_count,
        first_turn_outcome.streamed_fragment_count,
        follow_up_outcome.streamed_fragment_count,
    );
}

struct FollowUpJourneyRequestOutcome {
    finish_reason: Option<String>,
    model_text: String,
    error_text: String,
    generated_token_count: u64,
    streamed_fragment_count: u64,
}

async fn execute_follow_up_journey_request(
    openai_client: &LocalOpenAiClient,
    request_label: &'static str,
    completion_request: Value,
) -> FollowUpJourneyRequestOutcome {
    eprintln!("{LOG_MARKER} request={request_label} status=start");
    let mut outcome = FollowUpJourneyRequestOutcome {
        finish_reason: None,
        model_text: String::new(),
        error_text: String::new(),
        generated_token_count: 0,
        streamed_fragment_count: 0,
    };
    let mut streamed_completion = match openai_client
        .create_streaming_chat_completion(&completion_request)
        .await
    {
        Ok(streamed_completion) => streamed_completion,
        Err(request_error) => panic!(
            "{request_label} request must be admitted and start streaming; the admission \
             rejection surfaces here verbatim: {request_error}"
        ),
    };
    while let Some(stream_item) = streamed_completion.next().await {
        let stream_chunk = match stream_item {
            Ok(stream_chunk) => stream_chunk,
            Err(stream_error) => {
                outcome.error_text.push_str(&stream_error.to_string());
                break;
            }
        };
        if let Some(error_value) = stream_chunk.get("error")
            && !error_value.is_null()
        {
            outcome.error_text.push_str(&error_value.to_string());
        }
        for choice in stream_chunk["choices"].as_array().into_iter().flatten() {
            if let Some(content_fragment) = choice["delta"]["content"].as_str() {
                outcome.model_text.push_str(content_fragment);
                outcome.streamed_fragment_count += 1;
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                outcome.finish_reason = Some(reason.to_owned());
            }
        }
        // The streaming path does not always carry a usage block, so the
        // server-reported token count may stay zero; the fragment count stands
        // in for it in the summary. The server number wins when it appears.
        if let Some(completion_tokens) = stream_chunk["usage"]["completion_tokens"].as_u64() {
            outcome.generated_token_count = completion_tokens;
        }
    }
    eprintln!(
        "{LOG_MARKER} request={request_label} status=summary finish_reason={} generated_tokens={} streamed_fragments={} error_text={}",
        outcome.finish_reason.as_deref().unwrap_or("none"),
        outcome.generated_token_count,
        outcome.streamed_fragment_count,
        outcome.error_text,
    );
    outcome
}

fn assert_completed_follow_up_journey_request(
    request_label: &str,
    request_outcome: &FollowUpJourneyRequestOutcome,
) {
    assert!(
        !request_outcome.error_text.contains(REJECTION_SUBSTRING),
        "{request_label} must never surface the poisoned-reserve admission rejection: {}",
        request_outcome.error_text
    );
    assert!(
        request_outcome.error_text.is_empty(),
        "{request_label} stream must complete without errors: {}",
        request_outcome.error_text
    );
    assert!(
        matches!(
            request_outcome.finish_reason.as_deref(),
            Some("stop" | "length")
        ),
        "{request_label} must finish through the public stream; finish_reason={finish_reason:?}",
        finish_reason = request_outcome.finish_reason
    );
    assert!(
        !request_outcome.model_text.trim().is_empty(),
        "{request_label} must produce visible model text"
    );
}

fn write_follow_up_admission_config(
    isolated_worker_home: &Path,
    model_directory: &Path,
    ceiling_gb: u64,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the follow-up-admission configuration directory should be created");
    let configuration_document = json!({
        "model_directories": [acceptance_model_scan_root(model_directory)],
        "maximum_mlx_memory_gb": ceiling_gb,
        "max_output_tokens": MAXIMUM_OUTPUT_TOKEN_COUNT,
        "persistent_prompt_cache_enabled": true,
        "performance_attribution_enabled": true,
        "logging": {
            "level": "debug",
            "retained_files": 2,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": RESIDENT_CHUNK_TOKENS,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": SSD_STREAMING_CHUNK_TOKENS,
            "prefill_graph_submission_layer_interval": 0,
            "experimental_ssd_paging_prefill_graph_submission_layer_interval": 1,
            "experimental_ssd_paging_generation_graph_submission_layer_interval": 1,
        },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the follow-up-admission configuration should serialize"),
    )
    .expect("the follow-up-admission configuration should be written");
}
