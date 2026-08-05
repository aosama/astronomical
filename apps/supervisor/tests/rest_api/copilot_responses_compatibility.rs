use std::{future::Future, pin::Pin};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationCompletionReason, ChatMessage, ChatModelCapabilities,
    MtpRuntimeState,
};
use astronomical_supervisor::{
    ChatGenerationExecutor, ChatGenerationStreamEvent, GenerationStartError, WorkerHealthSnapshot,
    build_application,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt;

const MODEL_ID: &str = "astronomical/copilot-responses-compatibility-model";

#[tokio::test]
async fn should_complete_a_copilot_stream_with_visible_reasoning_and_supported_tool_schemas() {
    let (generation_command_sender, mut generation_command_receiver) = mpsc::unbounded_channel();
    let application = build_application(CopilotResponsesExecutor::with_command_capture(
        vec![
            ChatGenerationStreamEvent::ReasoningFragment("Inspecting locally.".to_owned()),
            ChatGenerationStreamEvent::TextFragment("ASTRONOMICAL_OK".to_owned()),
            completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
        ],
        generation_command_sender,
    ));
    let copilot_request_body = format!(
        r#"{{
            "model":"{MODEL_ID}",
            "input":[{{"role":"user","content":[{{"type":"input_text","text":"Reply exactly ASTRONOMICAL_OK"}}],"type":"message"}}],
            "tools":[
                {{"type":"function","name":"grep","parameters":{{"type":"object","properties":{{"paths":{{"anyOf":[{{"type":"string"}},{{"type":"array","items":{{"type":"string"}}}}]}}}}}},"strict":false}},
                {{"type":"function","name":"glob","parameters":{{"type":"object","properties":{{"paths":{{"anyOf":[{{"type":"string"}},{{"type":"array","items":{{"type":"string"}}}}]}}}}}},"strict":false}},
                {{"type":"function","name":"open_canvas","parameters":{{"type":"object","properties":{{"input":{{"description":"Canvas open input matching the canvas input schema"}}}}}},"strict":false}},
                {{"type":"function","name":"invoke_canvas_action","parameters":{{"type":"object","properties":{{"input":{{"description":"Action input matching the action input schema"}}}}}},"strict":false}}
            ],
            "parallel_tool_calls":true,
            "reasoning":{{"effort":"medium"}},
            "prompt_cache_key":"copilot-session",
            "max_output_tokens":20480,
            "store":false,
            "stream":true,
            "include":["reasoning.encrypted_content"]
        }}"#
    );

    let http_response = post_responses_request(application, copilot_request_body).await;

    assert_eq!(http_response.status(), StatusCode::OK);
    let response_text = read_response_text(http_response).await;
    assert!(response_text.contains("event: response.reasoning_summary_text.delta"));
    assert!(response_text.contains(r#""delta":"Inspecting locally.""#));
    assert!(response_text.contains(r#""type":"summary_text","text":"Inspecting locally.""#));
    assert!(!response_text.contains("encrypted_content"));
    assert!(response_text.contains(r#""delta":"ASTRONOMICAL_OK""#));
    assert!(response_text.contains("event: response.completed"));
    let generation_command = generation_command_receiver
        .try_recv()
        .expect("the Copilot request should reach worker admission");
    assert_eq!(generation_command.tools.len(), 4);
    assert!(generation_command.tools.iter().any(|tool_definition| {
        tool_definition.name == "grep"
            && tool_definition.parameters_json.contains(
                r#""anyOf":[{"type":"string"},{"items":{"type":"string"},"type":"array"}]"#,
            )
    }));
    assert!(generation_command.tools.iter().any(|tool_definition| {
        tool_definition.name == "invoke_canvas_action"
            && tool_definition
                .parameters_json
                .contains(r#""description":"Action input matching the action input schema""#)
    }));
}

#[tokio::test]
async fn should_replay_a_copilot_function_result_through_the_responses_endpoint() {
    let first_application = build_application(CopilotResponsesExecutor::new(vec![
        ChatGenerationStreamEvent::ToolCall {
            tool_call_index: 0,
            function_name: "view".to_owned(),
            arguments_json: r#"{"path":"package.json"}"#.to_owned(),
        },
        completed_generation_event(ChatGenerationCompletionReason::ToolCalls),
    ]));
    let first_request_body = json!({
        "model": MODEL_ID,
        "input": "Read package.json.",
        "tools": [{
            "type": "function",
            "name": "view",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            },
            "strict": false
        }],
        "stream": true
    })
    .to_string();
    let first_http_response = post_responses_request(first_application, first_request_body).await;
    let first_response_text = read_response_text(first_http_response).await;
    let function_call_item = parse_sse_payloads(&first_response_text)
        .into_iter()
        .find_map(|event_payload| {
            (event_payload["type"] == "response.output_item.done"
                && event_payload["item"]["type"] == "function_call")
                .then(|| event_payload["item"].clone())
        })
        .expect("the first response should contain a completed function call");
    let function_call_id = function_call_item["call_id"]
        .as_str()
        .expect("the function call should expose a call_id")
        .to_owned();

    let (generation_command_sender, mut generation_command_receiver) = mpsc::unbounded_channel();
    let replay_application = build_application(CopilotResponsesExecutor::with_command_capture(
        vec![
            ChatGenerationStreamEvent::TextFragment("ASTRONOMICAL_TOOL_OK".to_owned()),
            completed_generation_event(ChatGenerationCompletionReason::EndOfSequence),
        ],
        generation_command_sender,
    ));
    let replay_request_body = json!({
        "model": MODEL_ID,
        "input": [
            {"role": "user", "content": "Read package.json."},
            function_call_item,
            {"type": "function_call_output", "call_id": function_call_id, "output": "{\"name\":\"copilot-byok\"}"}
        ],
        "tools": [{
            "type": "function",
            "name": "view",
            "parameters": {"type": "object"},
            "strict": false
        }],
        "stream": true
    })
    .to_string();

    let replay_http_response =
        post_responses_request(replay_application, replay_request_body).await;

    assert_eq!(replay_http_response.status(), StatusCode::OK);
    let replay_response_text = read_response_text(replay_http_response).await;
    assert!(replay_response_text.contains(r#""delta":"ASTRONOMICAL_TOOL_OK""#));
    assert!(replay_response_text.contains("event: response.completed"));
    let replay_generation_command = generation_command_receiver
        .try_recv()
        .expect("the replay request should reach worker admission");
    let [
        ChatMessage::User { .. },
        ChatMessage::Assistant { tool_calls, .. },
        ChatMessage::Tool {
            tool_call_id,
            content,
        },
    ] = replay_generation_command.messages.as_slice()
    else {
        panic!(
            "expected user, assistant function call, and tool output messages; got {:?}",
            replay_generation_command.messages
        );
    };
    assert_eq!(tool_calls[0].id, *tool_call_id);
    assert_eq!(content, r#"{"name":"copilot-byok"}"#);
}

fn completed_generation_event(
    completion_reason: ChatGenerationCompletionReason,
) -> ChatGenerationStreamEvent {
    ChatGenerationStreamEvent::Completed {
        prompt_token_count: 100,
        generated_token_count: 10,
        reasoning_token_count: 0,
        cached_token_count: 0,
        reason: completion_reason,
    }
}

async fn post_responses_request(
    application: axum::Router,
    request_body: String,
) -> axum::response::Response {
    application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body))
                .expect("the Copilot Responses request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response")
}

async fn read_response_text(http_response: axum::response::Response) -> String {
    let response_body = to_bytes(http_response.into_body(), 128 * 1024)
        .await
        .expect("the Responses body should be readable");
    String::from_utf8(response_body.to_vec()).expect("the Responses body should be UTF-8")
}

fn parse_sse_payloads(response_text: &str) -> Vec<Value> {
    response_text
        .lines()
        .filter_map(|response_line| response_line.strip_prefix("data: "))
        .map(|event_payload_json| {
            serde_json::from_str(event_payload_json).expect("each SSE data payload should be JSON")
        })
        .collect()
}

struct CopilotResponsesExecutor {
    stream_events: Vec<ChatGenerationStreamEvent>,
    generation_command_sender: Option<mpsc::UnboundedSender<ChatGenerationCommand>>,
}

impl CopilotResponsesExecutor {
    fn new(stream_events: Vec<ChatGenerationStreamEvent>) -> Self {
        Self {
            stream_events,
            generation_command_sender: None,
        }
    }

    fn with_command_capture(
        stream_events: Vec<ChatGenerationStreamEvent>,
        generation_command_sender: mpsc::UnboundedSender<ChatGenerationCommand>,
    ) -> Self {
        Self {
            stream_events,
            generation_command_sender: Some(generation_command_sender),
        }
    }
}

impl ChatGenerationExecutor for CopilotResponsesExecutor {
    fn start_chat_generation(
        &self,
        generation_command: ChatGenerationCommand,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        mpsc::Receiver<ChatGenerationStreamEvent>,
                        GenerationStartError,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            if let Some(generation_command_sender) = &self.generation_command_sender {
                generation_command_sender
                    .send(generation_command)
                    .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            }
            let (stream_event_sender, stream_event_receiver) =
                mpsc::channel(self.stream_events.len().max(1));
            for stream_event in &self.stream_events {
                stream_event_sender
                    .send(stream_event.clone())
                    .await
                    .map_err(|_| GenerationStartError::WorkerUnavailable)?;
            }
            Ok(stream_event_receiver)
        })
    }

    fn worker_health_snapshot(&self) -> WorkerHealthSnapshot {
        WorkerHealthSnapshot::ready_with_model(
            MODEL_ID.to_owned(),
            ChatModelCapabilities {
                supports_reasoning: true,
                supports_tool_calls: true,
                has_vision: true,
                max_input_tokens: 241_664,
                max_output_tokens: 20_480,
                context_window: 262_144,
            },
            MtpRuntimeState::Disabled,
            None,
        )
    }
}
