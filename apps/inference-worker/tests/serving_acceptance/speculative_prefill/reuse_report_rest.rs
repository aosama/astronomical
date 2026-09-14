use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};
use serde_json::{Value, json};

use crate::serving_acceptance::chat::openai_rest::{
    E2E_TIMEOUT, get_endpoint, launch_serving_rest_server_for_model, post_chat_completion,
    stop_serving_rest_server,
};
use crate::small_dense_model::configured_deployment_litmus_model;

const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_5000_romeo_and_juliet_words.txt");

// Issue #657: the completion report must expose SpecPrefill sparse-target restoration.
// A warm second request restores the selection-bound sparse target prefix plus the dense
// control-span block, but the OpenAI usage surface reports only the ordinary dense
// restore, so a ~97% reuse session looks like ~2% reuse to clients that read
// `prompt_tokens_details.cached_tokens`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "launches the production worker and public REST surface with configured SpecPrefill and persistent caching"]
async fn should_report_restored_prompt_work_including_sparse_target_state_on_warm_requests() {
    tokio::time::timeout(E2E_TIMEOUT, async {
        let target_model = configured_deployment_litmus_model();
        let target_model_id = target_model.model_id;
        let target_model_directory = target_model.model_directory;
        // Reusing the smallest artifact as its own drafter keeps tokenizer compatibility
        // and bounds two-model memory on laptops without a separately packaged drafter.
        let draft_model_id = target_model_id.clone();
        let draft_model_directory = target_model_directory.clone();
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
        let model_artifact_rest_server = launch_serving_rest_server_for_model(
            &target_model_id,
            target_model_directory.clone(),
            Some(isolated_worker_home.path()),
            Some(performance_log_directory.path()),
        )
        .await;
        let server_address = model_artifact_rest_server.server_address;

        eprintln!("[specprefill-reuse-report-journey] status=progress phase=cold_request");
        let cold_request_body = warm_reuse_request_body(&target_model_id, &target_model_directory);
        let cold_response = post_chat_completion(server_address, cold_request_body.clone()).await;
        let cold_document = parse_http_json_response(&cold_response);
        assert_cold_request_completed(&cold_document);

        eprintln!("[specprefill-reuse-report-journey] status=progress phase=warm_request");
        // The warm request appends one assistant turn and one user turn to the same
        // conversation, so every earlier token is reusable prefix work.
        let warm_request_body = append_follow_up_turns(&cold_request_body);
        let warm_response = post_chat_completion(server_address, warm_request_body).await;
        let warm_document = parse_http_json_response(&warm_response);
        assert_cold_request_completed(&warm_document);

        let warm_usage = &warm_document["usage"];
        let reported_cached_tokens = warm_usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or_else(|| {
                panic!(
                    "the warm request usage should report prompt_tokens_details.cached_tokens: {warm_response}"
                )
            });
        let prompt_tokens = warm_usage["prompt_tokens"]
            .as_u64()
            .expect("the warm request usage should report prompt_tokens");

        // The reuse evidence lives in /v1/cache/stats regardless of what the usage
        // surface reports, so this endpoint anchors the expected magnitude.
        let cache_stats_response = get_endpoint(server_address, "/v1/cache/stats").await;
        let cache_stats_document = parse_http_json_response(&cache_stats_response);
        let target_restored = cache_stats_document["speculative_prefill_cache_efficacy"]
            ["target"]["restored_token_count"]
            .as_u64()
            .expect("cache stats should report SpecPrefill target restored tokens");

        eprintln!(
            "[specprefill-reuse-report-journey] status=reuse_evidence prompt_tokens={prompt_tokens} reported_cached_tokens={reported_cached_tokens} target_restored={target_restored}"
        );

        // The warm request must genuinely reuse most of its prompt through the
        // SpecPrefill sparse-target restore.
        assert!(
            target_restored > 0,
            "the warm request should restore SpecPrefill sparse target state",
        );

        // The defect under test: the usage-reported cached tokens must reflect the
        // combined dense + sparse restored prefix. Before the fix the warm request
        // reported no cached tokens at all (the sparse restore replaces dense state),
        // so a ~99%-reuse request looked like 0% reuse to usage consumers.
        assert!(
            reported_cached_tokens >= prompt_tokens.saturating_sub(4_096),
            "issue #657: usage cached_tokens ({reported_cached_tokens}) must include the \
             SpecPrefill sparse-target restored prefix; the warm prompt is {prompt_tokens} tokens",
        );

        stop_serving_rest_server(model_artifact_rest_server).await;
        eprintln!("[specprefill-reuse-report-journey] status=success");
    })
    .await
    .expect("the public SpecPrefill reuse-report journey should finish within 115 seconds");
}

fn assert_cold_request_completed(response_document: &Value) {
    let finish_reason = response_document["choices"][0]["finish_reason"]
        .as_str()
        .expect("the public response should report a completion reason");
    assert!(matches!(finish_reason, "stop" | "tool_calls" | "length"));
}

fn append_follow_up_turns(cold_request_body: &str) -> String {
    let mut warm_document: Value = serde_json::from_str(cold_request_body)
        .expect("the cold request body should parse as JSON");
    let messages = warm_document["messages"]
        .as_array_mut()
        .expect("the cold request body should contain messages");
    messages.push(json!({
        "role": "assistant",
        "content": "Recorded the central conflict and its tragic outcome.",
    }));
    messages.push(json!({
        "role": "user",
        "content": "Now summarize that conflict in one sentence without calling any tool.",
    }));
    warm_document.to_string()
}

fn warm_reuse_request_body(target_model_id: &str, target_model_directory: &Path) -> String {
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
    // Two source repetitions keep the prompt above the configured SpecPrefill
    // minimum so the sparse path engages on both requests.
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
                    request_id: RequestId::new(95_701),
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
                    qwen_thinking_channel_seed: None,
                    structured_generation: None,
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
        "[specprefill-reuse-report-journey] status=prompt_sized prompt_tokens={selected_prompt_token_count}"
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
