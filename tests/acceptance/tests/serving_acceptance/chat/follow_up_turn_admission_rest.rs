//! Public REST acceptance of the follow-up-turn admission journey (issue #690).
//!
//! Production failed exactly here: a client (Copilot CLI) sent a long first
//! turn, the worker admitted it, the chunked prefill taught the adaptive RAM
//! budget a chunk-shaped activation observation, and the follow-up turn —
//! same conversation, a few hundred more context tokens — was rejected five
//! times with `generation context exceeds available GPU wired memory` because
//! the admission path scaled that chunk observation by the total context
//! length. This journey replays that user journey through the public Chat
//! Completions TCP surface and requires both turns to complete.

use astronomical_model_serving::Qwen3_5ArtifactValidator;
use serde_json::json;
use serial_test::serial;
use std::fs;
use std::path::Path;

use super::openai_rest::{
    E2E_TIMEOUT, launch_serving_rest_server_for_model_with_memory_limit, post_chat_completion,
    stop_serving_rest_server,
};
use super::thinking_budget_support::{
    MAXIMUM_OUTPUT_TOKEN_COUNT, ROMEO_AND_JULIET_SOURCE, write_thinking_budget_acceptance_config,
};
use crate::small_dense_model;

const MAXIMUM_MODEL_TEST_MEMORY_BYTES: u64 = 36_000_000_000;

/// The complete Romeo and Juliet fixture spans several 2,048-token prefill
/// chunks, which is what teaches the adaptive budget its chunk-shaped
/// activation evidence. The production failure needed a multi-chunk prefill:
/// one chunk teaches the budget, and a follow-up turn whose total context
/// exceeds the highest measured bucket triggered the proportional projection.
/// The fixture is the mandated model test source.
fn long_romeo_and_juliet_prompt() -> String {
    ROMEO_AND_JULIET_SOURCE.to_owned()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model"]
#[serial]
async fn should_admit_the_follow_up_turn_after_a_long_successful_prefill() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let selected_model = small_dense_model::configured_deployment_litmus_model();
        // Validating up front turns a broken artifact into a named journey
        // failure instead of an opaque admission rejection inside the worker.
        Qwen3_5ArtifactValidator::new()
            .validate(
                &selected_model.model_directory,
                u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT),
            )
            .expect("the smallest configured Qwen3.5 artifact should validate");
        let isolated_worker_home = tempfile::tempdir()
            .expect("the follow-up-turn journey should create an isolated worker home");
        write_thinking_budget_acceptance_config(
            isolated_worker_home.path(),
            &selected_model.model_id,
            &selected_model.model_directory,
            MAXIMUM_OUTPUT_TOKEN_COUNT,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the follow-up-turn journey should create a performance-log directory");
        let rest_server = launch_serving_rest_server_for_model_with_memory_limit(
            &selected_model.model_id,
            selected_model.model_directory,
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
            Some(MAXIMUM_MODEL_TEST_MEMORY_BYTES),
        )
        .await;

        // Turn one: a multi-chunk prefill (several 2,048-token chunks). This
        // is the request that teaches the adaptive budget its chunk-shaped
        // activation evidence. A small output cap keeps the journey inside
        // the time bound; the memory-admission defect is exercised by the
        // prompt crossing multiple chunk boundaries, not the generation
        // length.
        let first_turn_prompt = long_romeo_and_juliet_prompt();
        eprintln!(
            "[follow-up-turn-admission] status=progress phase=first_turn model={} prompt_characters={}",
            selected_model.model_id,
            first_turn_prompt.len()
        );
        let first_turn_response = post_chat_completion(
            rest_server.server_address,
            json!({
                "model": selected_model.model_id,
                "messages": [{"role": "user", "content": first_turn_prompt}],
                "max_tokens": 512,
                "stream": false,
            })
            .to_string(),
        )
        .await;
        assert!(
            first_turn_response.starts_with("HTTP/1.1 200 OK"),
            "the first long turn must be admitted and complete: {}",
            first_turn_response
        );

        // Turn two: the follow-up in the same conversation. Its total context
        // exceeds turn one's, which is the exact shape that poisoned admission
        // in production. The journey requires admission and a clean completion,
        // never `generation context exceeds available GPU wired memory`.
        let follow_up_response = post_chat_completion(
            rest_server.server_address,
            json!({
                "model": selected_model.model_id,
                "messages": [
                    {"role": "user", "content": first_turn_prompt},
                    {"role": "assistant", "content": "Understood, I have the excerpt in mind."},
                    {"role": "user", "content": "In one sentence: who wrote the excerpt and what is it about?"}
                ],
                "max_tokens": 512,
                "stream": false,
            })
            .to_string(),
        )
        .await;
        stop_serving_rest_server(rest_server).await;
        assert!(
            follow_up_response.starts_with("HTTP/1.1 200 OK"),
            "the follow-up turn must be admitted after the long successful prefill: {}",
            follow_up_response
        );
        assert!(
            !follow_up_response.contains("generation context exceeds available GPU wired memory"),
            "the poisoned-reserve rejection must never surface on the follow-up turn"
        );

        let follow_up_completion = serde_json::from_str::<serde_json::Value>(
            follow_up_response
                .split_once("\r\n\r\n")
                .map(|(_, body)| body)
                .unwrap_or(""),
        )
        .expect("the follow-up completion should carry a JSON body");
        let follow_up_message = &follow_up_completion["choices"][0]["message"];
        let follow_up_text = follow_up_message["content"].as_str().unwrap_or("");
        let follow_up_reasoning = follow_up_message["reasoning_content"].as_str().unwrap_or("");
        assert!(
            !(follow_up_text.trim().is_empty() && follow_up_reasoning.trim().is_empty()),
            "the follow-up turn must produce completion output"
        );
        eprintln!(
            "[follow-up-turn-admission] status=success follow_up_characters={}",
            follow_up_text.len() + follow_up_reasoning.len()
        );
    })
    .await
    .expect("the follow-up-turn admission journey must finish within 115 seconds");
}

/// The smaller-variant companion of the paged-MoE regression journey: a dense
/// model never pages experts, so the admission operation bound must resolve to
/// the resident chunk even when a wider SSD-streaming chunk is configured.
/// The pre-fix code resolved the resident mode's promise at the larger paged
/// scope and inflated the reserve by the chunk ratio (measured 2026-10-10 on
/// the paged artifact). Dense models prove the mode-bound selection without
/// the multi-gigabyte MoE fixture.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production REST surface and smallest configured Qwen3.5 model with a wider SSD-streaming chunk configured"]
#[serial]
async fn should_admit_the_follow_up_turn_when_the_ssd_streaming_chunk_is_wider() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let selected_model = small_dense_model::configured_deployment_litmus_model();
        // Validating up front turns a broken artifact into a named journey
        // failure instead of an opaque admission rejection inside the worker.
        Qwen3_5ArtifactValidator::new()
            .validate(
                &selected_model.model_directory,
                u32::from(MAXIMUM_OUTPUT_TOKEN_COUNT),
            )
            .expect("the smallest configured Qwen3.5 artifact should validate");
        let isolated_worker_home = tempfile::tempdir()
            .expect("the follow-up-turn journey should create an isolated worker home");
        write_split_chunk_follow_up_admission_config(
            isolated_worker_home.path(),
            &selected_model.model_id,
            &selected_model.model_directory,
            MAXIMUM_OUTPUT_TOKEN_COUNT,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the follow-up-turn journey should create a performance-log directory");
        let rest_server = launch_serving_rest_server_for_model_with_memory_limit(
            &selected_model.model_id,
            selected_model.model_directory,
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
            Some(MAXIMUM_MODEL_TEST_MEMORY_BYTES),
        )
        .await;

        // Turn one: the multi-chunk prefill that teaches the budget its
        // chunk-shaped activation evidence at the resident operation scope.
        let first_turn_prompt = long_romeo_and_juliet_prompt();
        let first_turn_response = post_chat_completion(
            rest_server.server_address,
            json!({
                "model": selected_model.model_id,
                "messages": [{"role": "user", "content": first_turn_prompt}],
                "max_tokens": 512,
                "stream": false,
            })
            .to_string(),
        )
        .await;
        assert!(
            first_turn_response.starts_with("HTTP/1.1 200 OK"),
            "the first long turn must be admitted with a wider SSD-streaming chunk configured: {}",
            first_turn_response
        );

        // Turn two: the same-conversation follow-up whose total context
        // exceeds turn one's, the shape that the pre-fix admission inflated.
        let follow_up_response = post_chat_completion(
            rest_server.server_address,
            json!({
                "model": selected_model.model_id,
                "messages": [
                    {"role": "user", "content": first_turn_prompt},
                    {"role": "assistant", "content": "Understood, I have the excerpt in mind."},
                    {"role": "user", "content": "In one sentence: who wrote the excerpt and what is it about?"}
                ],
                "max_tokens": 512,
                "stream": false,
            })
            .to_string(),
        )
        .await;
        stop_serving_rest_server(rest_server).await;
        assert!(
            follow_up_response.starts_with("HTTP/1.1 200 OK"),
            "the follow-up turn must be admitted when the resident mode's own operation \
             scope governs the activation reserve: {}",
            follow_up_response
        );
        assert!(
            !follow_up_response.contains("generation context exceeds available GPU wired memory"),
            "the poisoned-reserve rejection must never surface on the follow-up turn"
        );
    })
    .await
    .expect("the split-chunk follow-up-turn admission journey must finish within 115 seconds");
}

/// Writes the acceptance configuration with a wider SSD-streaming chunk so the
/// resident mode's admission bound is provably decoupled from the paged scope.
fn write_split_chunk_follow_up_admission_config(
    isolated_worker_home: &Path,
    model_id: &str,
    model_directory: &Path,
    maximum_output_tokens: u16,
) {
    let configuration_directory = isolated_worker_home.join(".astronomical-dev");
    fs::create_dir(&configuration_directory)
        .expect("the isolated Astronomical configuration directory should be created");
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": { "model_directories": [model_directory] },
        "prompt_cache": { "enabled": false, "maximum_size_gb": 50 },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 2_048,
            "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 4_096,
        },
        "models": {
            (model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": maximum_output_tokens,
                },
            },
        },
        "diagnostics": { "performance_attribution_enabled": true },
    });
    fs::write(
        configuration_directory.join("config.json"),
        serde_json::to_vec_pretty(&configuration_document)
            .expect("the split-chunk acceptance configuration should serialize"),
    )
    .expect("the split-chunk acceptance configuration should be written");
}
