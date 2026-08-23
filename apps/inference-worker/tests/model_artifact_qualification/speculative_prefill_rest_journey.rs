use std::path::{Path, PathBuf};

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::{Value, json};

use super::model_artifact_rest_qualification::{
    E2E_TIMEOUT, get_endpoint, launch_model_artifact_rest_server_for_model, post_chat_completion,
    stop_model_artifact_rest_server,
};

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production worker and public REST surface with configured SpecPrefill and persistent caching"]
async fn should_complete_the_cold_tool_journey_through_real_config_worker_and_rest_boundaries() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let development_home_directory = std::env::var_os("HOME")
            .map(PathBuf::from)
            .expect("HOME should resolve the Development Astronomical configuration");
        let development_config =
            AstronomicalConfig::load_from_development_home_directory(&development_home_directory)
                .expect("the Development Astronomical configuration should load");
        let target_model_id =
            crate::common::ORNITH_MODEL_ARTIFACT_QUALIFICATION_MODEL_ID.to_owned();
        let target_model_context_window = crate::common::configured_discovered_models()
            .into_iter()
            .find(|model| model.model_id == target_model_id)
            .and_then(|model| {
                crate::common::chat_capabilities(&model)
                    .map(|chat_capabilities| chat_capabilities.context_window)
            })
            .expect("the qualification target should be discovered as a chat model");
        let configured_target_policy = development_config
            .resolved_model_config(&target_model_id, target_model_context_window)
            .expect("the Development target policy should resolve");
        let configured_speculative_prefill = configured_target_policy
            .speculative_prefill()
            .expect("the Development target should configure a SpecPrefill drafter");
        let draft_model_id = configured_speculative_prefill
            .draft_model_id()
            .expect("the qualification requires a configured SpecPrefill drafter")
            .to_owned();
        let target_model_directory =
            crate::common::configured_model_artifact_directory_by_id(&target_model_id);
        let draft_model_directory =
            crate::common::configured_model_artifact_directory_by_id(&draft_model_id);
        let isolated_worker_home = tempfile::tempdir()
            .expect("the public SpecPrefill journey should create an isolated worker home");
        write_enabled_speculative_prefill_config(
            isolated_worker_home.path(),
            &target_model_id,
            &target_model_directory,
            &draft_model_id,
            &draft_model_directory,
        );
        let performance_log_directory = tempfile::tempdir()
            .expect("the public SpecPrefill journey should create a performance-log directory");
        let model_artifact_rest_server = launch_model_artifact_rest_server_for_model(
            &target_model_id,
            target_model_directory.clone(),
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
        )
        .await;
        let server_address = model_artifact_rest_server.server_address;

        eprintln!("[speculative-prefill-rest-journey] status=progress phase=cold_tool_request");
        let cold_tool_request_body = cold_tool_request_body(
            &target_model_id,
            &target_model_directory,
        );
        let chat_response = post_chat_completion(
            server_address,
            cold_tool_request_body,
        )
        .await;
        let response_document = parse_http_json_response(&chat_response);
        let finish_reason = response_document["choices"][0]["finish_reason"]
            .as_str()
            .expect("the public response should report a completion reason");
        assert!(matches!(finish_reason, "stop" | "tool_calls"));
        let response_message = &response_document["choices"][0]["message"];
        assert!(
            response_message["content"]
                .as_str()
                .is_some_and(|model_visible_content| !model_visible_content.trim().is_empty())
                || response_message["tool_calls"]
                    .as_array()
                    .is_some_and(|tool_calls| !tool_calls.is_empty()),
            "the public tool-bearing request should return model-visible output",
        );

        let status_response = get_endpoint(server_address, "/v1/status").await;
        let status_document = parse_http_json_response(&status_response);
        assert_eq!(status_document["speculative_prefill_enabled"], true);
        assert_eq!(status_document["speculative_prefill_runtime_state"], "active");
        assert_eq!(
            status_document["speculative_prefill_target_model_id"],
            target_model_id,
        );
        assert_eq!(
            status_document["speculative_prefill_draft_model_id"],
            draft_model_id,
        );

        let cache_stats_response = get_endpoint(server_address, "/v1/cache/stats").await;
        let cache_stats_document = parse_http_json_response(&cache_stats_response);
        assert!(
            cache_stats_document["speculative_prefill_cache_efficacy"]["target"]
                ["eligible_token_count"]
                .as_u64()
                .is_some_and(|eligible_token_count| eligible_token_count > 0)
        );
        assert!(
            cache_stats_document["speculative_prefill_cache_efficacy"]["drafter"]
                ["eligible_token_count"]
                .as_u64()
                .is_some_and(|eligible_token_count| eligible_token_count > 0)
        );
        assert_eq!(
            cache_stats_document["speculative_prefill_cache_efficacy"]["target"]
                ["restored_token_count"],
            0,
        );
        assert_eq!(
            cache_stats_document["speculative_prefill_cache_efficacy"]["drafter"]
                ["restored_token_count"],
            0,
        );
        assert!(
            cache_file_count(&isolated_worker_home.path().join(".astronomical-dev/cache")) > 0,
            "the cold public request must publish cache-eligible state to SSD",
        );

        stop_model_artifact_rest_server(model_artifact_rest_server).await;
        eprintln!("[speculative-prefill-rest-journey] status=success");
    })
    .await
    .expect("the public cold SpecPrefill tool journey should finish within 115 seconds");
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
    let configuration_document = json!({
        "$schema": "./astronomical-config.schema.json",
        "schema_version": 1,
        "runtime": {
            "model_directories": [target_model_directory, draft_model_directory],
        },
        "prompt_cache": {
            "enabled": true,
            "maximum_size_gb": 50,
        },
        "chunking": {
            "fixed_prompt_processing_chunk_size_tokens": 32,
        },
        "models": {
            (target_model_id): {
                "generation_defaults": {
                    "maximum_output_tokens": 256,
                },
                "acceleration": {
                    "speculative_prefill": {
                        "draft_model_id": draft_model_id,
                        "minimum_prompt_tokens": 2048,
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

fn cold_tool_request_body(target_model_id: &str, target_model_directory: &Path) -> String {
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(target_model_directory, 256)
        .expect("the public journey target artifact should validate for prompt sizing");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the public journey tokenizer should load for prompt sizing");
    let declared_tool = ChatToolDefinition {
        name: "record_literary_analysis".to_owned(),
        description: Some("Record a structured literary analysis.".to_owned()),
        parameters_json: r#"{"type":"object","properties":{"central_conflict":{"type":"string"},"outcome":{"type":"string","enum":["tragic","comic"]}},"required":["central_conflict","outcome"],"additionalProperties":false}"#.to_owned(),
    };
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
                    request_id: RequestId::new(95_699),
                    model: target_model_id.to_owned(),
                    messages: vec![
                        ChatMessage::System {
                            content: "Use the declared tool and return its required fields."
                                .to_owned(),
                        },
                        ChatMessage::User {
                            content: candidate_source_message.clone(),
                            images: Vec::new(),
                        },
                        ChatMessage::Assistant {
                            content: Some("I have reviewed the supplied source material.".to_owned()),
                            reasoning_content: None,
                            tool_calls: Vec::new(),
                        },
                        ChatMessage::User {
                            content: "Call record_literary_analysis now. Record the play's central conflict and classify its outcome as tragic. Return only the declared tool call with central_conflict and outcome."
                                .to_owned(),
                            images: Vec::new(),
                        },
                    ],
                    tools: vec![declared_tool.clone()],
                    tool_choice: ChatToolChoice::Auto,
                    settings: ChatGenerationSettings {
                        max_output_tokens: 256,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                },
                false,
            )
            .expect("the public tool prompt candidate should prepare")
            .input_token_ids()
            .len();
        if candidate_prompt_token_count <= 8_192 {
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
        selected_prompt_token_count >= 8_000,
        "the public Romeo and Juliet request should remain representative after token-aware sizing",
    );
    eprintln!(
        "[speculative-prefill-rest-journey] status=prompt_sized prompt_tokens={selected_prompt_token_count}"
    );
    json!({
        "model": target_model_id,
        "messages": [
            {
                "role": "system",
                "content": "Use the declared tool and return its required fields.",
            },
            {
                "role": "user",
                "content": selected_user_prompt,
            },
            {
                "role": "assistant",
                "content": "I have reviewed the supplied source material.",
            },
            {
                "role": "user",
                "content": "Call record_literary_analysis now. Record the play's central conflict and classify its outcome as tragic. Return only the declared tool call with central_conflict and outcome.",
            },
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "record_literary_analysis",
                "description": "Record a structured literary analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "central_conflict": {"type": "string"},
                        "outcome": {"type": "string", "enum": ["tragic", "comic"]},
                    },
                    "required": ["central_conflict", "outcome"],
                    "additionalProperties": false,
                },
            },
        }],
        "tool_choice": "auto",
        "stream": false,
        "temperature": 0,
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

fn cache_file_count(cache_root_directory: &Path) -> usize {
    let mut pending_cache_directories = vec![cache_root_directory.to_path_buf()];
    let mut cache_file_count = 0_usize;
    while let Some(cache_directory) = pending_cache_directories.pop() {
        let Ok(cache_directory_entries) = std::fs::read_dir(cache_directory) else {
            continue;
        };
        for cache_directory_entry in cache_directory_entries.flatten() {
            let cache_entry_path = cache_directory_entry.path();
            if cache_entry_path.is_dir() {
                pending_cache_directories.push(cache_entry_path);
            } else if cache_entry_path.is_file() {
                cache_file_count += 1;
            }
        }
    }
    cache_file_count
}
