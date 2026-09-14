use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::{Value, json};

use crate::serving_acceptance::chat::openai_rest::{
    E2E_TIMEOUT, get_endpoint, post_chat_completion,
};
use crate::small_dense_model::configured_deployment_litmus_model;
use crate::support::serving_rest::{
    ServingRestServer, launch_serving_rest_server_for_model, stop_serving_rest_server,
};

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

/// The journeys pin a small cache block so a modest appended conversation tail
/// still completes whole SSD blocks. 512 is a multiple of the 256-token
/// persistence alignment and stays far below the model's context window.
const PINNED_CACHE_BLOCK_TOKEN_COUNT: u32 = 512;

/// Cold prompts must engage SpecPrefill sparse capture (above the minimum), while
/// the appended warm tail must fall below the same minimum so the request runs
/// target-only and exercises exactly the dense-tail publication defect (#659).
const SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS: u32 = 2_048;
const COLD_PROMPT_TARGET_TOKEN_COUNT: usize = 2_600;
const APPENDED_TAIL_TARGET_TOKEN_COUNT: usize = 1_200;

// Issue #659: when SpecPrefill sparse target state restores a prompt prefix,
// dense prompt-cache capture is disabled for the whole request. The appended
// conversation tail is processed with ordinary dense forwards but never
// published, so the SSD cache stays pinned at the initial sparse boundary and
// every request re-prefills the entire growing tail.
//
// This journey covers acceptance criterion 1 (tail publication): a cold request
// captures the sparse prefix, the warm request's target-only dense tail must be
// published as SSD blocks, and the cache must visibly grow.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production worker and public REST surface with configured SpecPrefill and persistent caching"]
async fn should_publish_dense_ssd_blocks_for_the_target_only_conversation_tail() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let (model_artifact_rest_server, _isolated_worker_home, cold_request_body, cache_stats_after_cold) =
            launch_specprefill_worker_with_cold_request().await;
        let server_address = model_artifact_rest_server.server_address;

        eprintln!("[specprefill-tail-publication-journey] status=progress phase=warm_request");
        let warm_request_body = append_romeo_and_juliet_tail(
            &cold_request_body,
            APPENDED_TAIL_TARGET_TOKEN_COUNT,
            "Summarize the appended continuation of the play in two sentences.",
        );
        let warm_response = post_chat_completion(server_address, warm_request_body).await;
        let warm_document = parse_http_json_response(&warm_response);
        assert_request_completed(&warm_document, &warm_response);

        let block_token_count = cache_stats_after_cold["persistent_prompt_cache_block_token_count"]
            .as_u64()
            .unwrap_or_else(|| {
                panic!("cache stats should report block_token_count: {cache_stats_after_cold}")
            });

        let cache_stats_after_warm = read_cache_stats(server_address).await;
        let warm_sequence_state_block_count =
            cache_stats_after_warm["persistent_prompt_cache_sequence_state_block_count"]
                .as_u64()
                .unwrap_or_else(|| {
                    panic!(
                        "cache stats should report the sequence-state block count: {cache_stats_after_warm}"
                    )
                });
        let cold_sequence_state_block_count =
            cache_stats_after_cold["persistent_prompt_cache_sequence_state_block_count"]
                .as_u64()
                .unwrap_or_else(|| {
                    panic!(
                        "cache stats should report the sequence-state block count: {cache_stats_after_cold}"
                    )
                });
        eprintln!(
            "[specprefill-tail-publication-journey] status=publication_evidence block_tokens={block_token_count} cold_blocks={cold_sequence_state_block_count} warm_blocks={warm_sequence_state_block_count}"
        );

        // The defect under test: the warm request's dense tail must publish SSD
        // blocks. Before the fix the sequence-state block count stays pinned at
        // the cold capture because the request-scope flag disables dense capture
        // for the entire request.
        assert!(
            warm_sequence_state_block_count > cold_sequence_state_block_count,
            "issue #659: the target-only conversation tail must publish dense SSD blocks \
             (after cold={cold_sequence_state_block_count}, after warm={warm_sequence_state_block_count}, \
             block_tokens={block_token_count})"
        );

        stop_serving_rest_server(model_artifact_rest_server).await;
        eprintln!("[specprefill-tail-publication-journey] status=success");
    })
    .await
    .expect("the SpecPrefill dense-tail publication journey should finish within 115 seconds");
}

// Issue #659 acceptance criterion 2 (tail restore): a third request repeating
// the warm conversation must restore the union of the sparse prefix and the
// published dense tail, so its reported cached tokens exceed the sparse prefix
// alone. Before the fix the restore stays pinned at the sparse boundary and the
// dense tail is re-prefilled every request.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production worker and public REST surface with configured SpecPrefill and persistent caching"]
async fn should_restore_the_sparse_prefix_plus_published_dense_tail_union() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let (model_artifact_rest_server, _isolated_worker_home, cold_request_body, _cache_stats_after_cold) =
            launch_specprefill_worker_with_cold_request().await;
        let server_address = model_artifact_rest_server.server_address;

        eprintln!("[specprefill-tail-restore-journey] status=progress phase=warm_request");
        let warm_request_body = append_romeo_and_juliet_tail(
            &cold_request_body,
            APPENDED_TAIL_TARGET_TOKEN_COUNT,
            "Summarize the appended continuation of the play in two sentences.",
        );
        let warm_response = post_chat_completion(server_address, warm_request_body.clone()).await;
        let warm_document = parse_http_json_response(&warm_response);
        assert_request_completed(&warm_document, &warm_response);

        eprintln!("[specprefill-tail-restore-journey] status=progress phase=repeat_request");
        let repeat_response = post_chat_completion(server_address, warm_request_body).await;
        let repeat_document = parse_http_json_response(&repeat_response);
        assert_request_completed(&repeat_document, &repeat_response);

        let repeat_cached_tokens = reported_cached_tokens(&repeat_document, &repeat_response);
        let warm_cached_tokens = reported_cached_tokens(&warm_document, &warm_response);
        eprintln!(
            "[specprefill-tail-restore-journey] status=restore_evidence warm_cached={warm_cached_tokens} repeat_cached={repeat_cached_tokens}"
        );

        // The repeat request must restore more than the warm request did: the
        // dense tail blocks published by the warm request must also restore.
        // Before the fix both requests report exactly the sparse boundary, so
        // the repeat request can never exceed the warm one.
        assert!(
            repeat_cached_tokens > warm_cached_tokens,
            "issue #659: the repeat request must restore the sparse prefix plus the published \
             dense tail (repeat={repeat_cached_tokens} must exceed warm={warm_cached_tokens})"
        );

        stop_serving_rest_server(model_artifact_rest_server).await;
        eprintln!("[specprefill-tail-restore-journey] status=success");
    })
    .await
    .expect("the SpecPrefill dense-tail restore journey should finish within 115 seconds");
}

/// Launches one SpecPrefill-enabled worker, sends the cold request that engages
/// sparse capture, and returns the server handle, the isolated home (kept alive
/// for the whole journey because the worker reads its config and writes its
/// cache inside it), and the cold request body for conversation extension.
async fn launch_specprefill_worker_with_cold_request() -> (
    ServingRestServer,
    tempfile::TempDir,
    String,
    serde_json::Value,
) {
    let target_model = configured_deployment_litmus_model();
    let target_model_id = target_model.model_id;
    let target_model_directory = target_model.model_directory;
    // Reusing the smallest artifact as its own drafter keeps tokenizer compatibility
    // and bounds two-model memory on laptops without a separately packaged drafter.
    let draft_model_id = target_model_id.clone();
    let draft_model_directory = target_model_directory.clone();
    let isolated_worker_home = tempfile::tempdir()
        .expect("the SpecPrefill dense-tail journey should create an isolated worker home");
    write_enabled_speculative_prefill_config(
        isolated_worker_home.path(),
        &target_model_id,
        &target_model_directory,
        &draft_model_id,
        &draft_model_directory,
    );
    let performance_log_directory = tempfile::tempdir()
        .expect("the SpecPrefill dense-tail journey should create a performance-log directory");
    let model_artifact_rest_server = launch_serving_rest_server_for_model(
        &target_model_id,
        target_model_directory.clone(),
        Some(isolated_worker_home.path()),
        Some(performance_log_directory.path()),
    )
    .await;

    eprintln!("[specprefill-tail-journeys] status=progress phase=cold_request");
    let cold_request_body = cold_request_body(
        &target_model_id,
        &target_model_directory,
        COLD_PROMPT_TARGET_TOKEN_COUNT,
    );
    let cold_response = post_chat_completion(
        model_artifact_rest_server.server_address,
        cold_request_body.clone(),
    )
    .await;
    let cold_document = parse_http_json_response(&cold_response);
    assert_request_completed(&cold_document, &cold_response);
    let cold_prompt_tokens = cold_document["usage"]["prompt_tokens"]
        .as_u64()
        .expect("the cold request usage should report prompt_tokens");
    assert!(
        cold_prompt_tokens >= u64::from(SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS),
        "the cold prompt must engage SpecPrefill: prompt_tokens={cold_prompt_tokens} \
         minimum={SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS}",
    );
    // Snapshot the cache before the warm request so the warm publication delta is
    // attributable to the target-only tail and not to the cold capture.
    let cache_stats_after_cold = read_cache_stats(model_artifact_rest_server.server_address).await;

    // The performance-log tempdir is only needed while the server runs, but the
    // isolated home must outlive it: the worker reads its config and writes its
    // SSD cache inside that path for every request in the journey.
    std::mem::forget(performance_log_directory);
    (
        model_artifact_rest_server,
        isolated_worker_home,
        cold_request_body,
        cache_stats_after_cold,
    )
}

fn reported_cached_tokens(response_document: &Value, raw_response: &str) -> u64 {
    response_document["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| {
            panic!(
                "the request usage should report prompt_tokens_details.cached_tokens: {raw_response}"
            )
        })
}

async fn read_cache_stats(server_address: std::net::SocketAddr) -> Value {
    let cache_stats_response = get_endpoint(server_address, "/v1/cache/stats").await;
    parse_http_json_response(&cache_stats_response)
}

fn assert_request_completed(response_document: &Value, raw_response: &str) {
    let finish_reason = response_document["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_else(|| {
            panic!("the public response should report a completion reason: {raw_response}")
        });
    assert!(
        matches!(finish_reason, "stop" | "tool_calls" | "length"),
        "the request should complete normally: finish_reason={finish_reason} response={raw_response}"
    );
}

fn append_romeo_and_juliet_tail(
    cold_request_body: &str,
    tail_target_token_count: usize,
    tail_instruction: &str,
) -> String {
    let mut warm_document: Value = serde_json::from_str(cold_request_body)
        .expect("the cold request body should parse as JSON");
    let messages = warm_document["messages"]
        .as_array_mut()
        .expect("the cold request body should contain messages");
    // A rough characters-to-tokens ratio keeps the appended tail comfortably
    // above one pinned block while remaining below the SpecPrefill minimum.
    let tail_source = ROMEO_AND_JULIET_SOURCE
        .chars()
        .take(tail_target_token_count.saturating_mul(3))
        .collect::<String>();
    messages.push(json!({
        "role": "assistant",
        "content": "I have reviewed the supplied source material.",
    }));
    messages.push(json!({
        "role": "user",
        "content": format!("Continuation of the source material:\n\n{tail_source}\n\n{tail_instruction}"),
    }));
    warm_document.to_string()
}

fn cold_request_body(
    target_model_id: &str,
    target_model_directory: &Path,
    target_prompt_token_count: usize,
) -> String {
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(target_model_directory, 256)
        .expect("the public journey target artifact should validate for prompt sizing");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the public journey tokenizer should load for prompt sizing");
    let repeated_source_material = ROMEO_AND_JULIET_SOURCE.repeat(2);
    let source_character_boundaries = repeated_source_material
        .char_indices()
        .map(|(byte_position, _source_character)| byte_position)
        .chain(std::iter::once(repeated_source_material.len()))
        .collect::<Vec<_>>();
    let mut lower_character_position = 0_usize;
    let mut upper_character_position = source_character_boundaries.len() - 1;
    let mut selected_user_prompt = String::new();
    let mut selected_prompt_token_count = 0_usize;
    while lower_character_position <= upper_character_position {
        let candidate_character_position =
            lower_character_position + (upper_character_position - lower_character_position) / 2;
        let candidate_source_end_byte_position =
            source_character_boundaries[candidate_character_position];
        let candidate_source_message = format!(
            "Romeo and Juliet source material for the requested analysis:\n\n{}",
            &repeated_source_material[..candidate_source_end_byte_position],
        );
        let candidate_prompt_token_count = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_801),
                    model: target_model_id.to_owned(),
                    messages: vec![
                        ChatMessage::System {
                            content: "Summarize the supplied source material faithfully."
                                .to_owned(),
                        },
                        ChatMessage::User {
                            content: candidate_source_message.clone(),
                            images: Vec::new(),
                        },
                    ],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: 256,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                    qwen_thinking_channel_seed: None,
                    structured_generation: None,
                },
                false,
            )
            .expect("the public prompt candidate should prepare")
            .input_token_ids()
            .len();
        if candidate_prompt_token_count <= target_prompt_token_count {
            selected_user_prompt = candidate_source_message;
            selected_prompt_token_count = candidate_prompt_token_count;
            lower_character_position = candidate_character_position.saturating_add(1);
        } else if candidate_character_position == 0 {
            break;
        } else {
            upper_character_position = candidate_character_position - 1;
        }
    }
    assert!(
        selected_prompt_token_count >= SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS as usize,
        "the cold prompt must stay above the SpecPrefill minimum after token-aware sizing: \
         prompt_tokens={selected_prompt_token_count}",
    );
    eprintln!(
        "[specprefill-tail-journeys] status=prompt_sized cold_prompt_tokens={selected_prompt_token_count}"
    );
    json!({
        "model": target_model_id,
        "messages": [
            {
                "role": "system",
                "content": "Summarize the supplied source material faithfully.",
            },
            {
                "role": "user",
                "content": selected_user_prompt,
            },
        ],
        "stream": false,
        "temperature": 1,
        "thinking_budget": 0,
        "max_tokens": 256,
    })
    .to_string()
}

fn parse_http_json_response(http_response: &str) -> Value {
    assert!(
        http_response.starts_with("HTTP/1.1 200 OK"),
        "unexpected HTTP response: {http_response}"
    );
    let (_, response_body) = http_response
        .split_once("\r\n\r\n")
        .expect("the HTTP response should contain a header/body boundary");
    serde_json::from_str(response_body).expect("the HTTP response body should contain JSON")
}

fn write_enabled_speculative_prefill_config(
    isolated_worker_home: &Path,
    target_model_id: &str,
    target_model_directory: &Path,
    draft_model_id: &str,
    draft_model_directory: &Path,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    std::fs::create_dir(&configuration_directory)
        .expect("the isolated Astronomical configuration directory should be created");
    let model_directories = if target_model_directory == draft_model_directory {
        vec![target_model_directory]
    } else {
        vec![target_model_directory, draft_model_directory]
    };
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": model_directories,
        },
        "prompt_cache": {
            "enabled": true,
            "maximum_size_gb": 50,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 32,
            "prompt_cache_block_tokens": PINNED_CACHE_BLOCK_TOKEN_COUNT,
        },
        "models": {
            (target_model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": 256,
                },
                "acceleration": {
                    "speculative_prefill": {
                        "draft_model_id": draft_model_id,
                        "minimum_prompt_tokens": SPECULATIVE_PREFILL_MINIMUM_PROMPT_TOKENS,
                        "keep_percentage": 20,
                    },
                },
            },
        },
        "diagnostics": {
            "performance_attribution_enabled": true,
        },
    });
    std::fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the isolated SpecPrefill configuration should serialize"),
    )
    .expect("the isolated SpecPrefill configuration should be written");
}
