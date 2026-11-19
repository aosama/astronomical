//! Unit tests for `crate::rest::openai::chat_translation`: how an
//! OpenAI chat completion request lowers into the worker's chat command.

use ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition, RequestId,
    StructuredGenerationConstraint,
};
use rest_contract::OpenAiChatCompletionRequest;

#[test]
fn should_lower_a_later_system_message_to_a_chronological_user_update() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Existing conversation context."},
            {"role": "system", "content": "A chronological policy update."},
            {"role": "user", "content": "hello"}
        ],
        "stream": true
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the chronological system-update request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(904), request)
            .expect("a later system message should lower into visible user text");

    assert_eq!(
        chat_command.messages,
        vec![
            ChatMessage::User {
                content: "Existing conversation context.\n<system-update>\nA chronological policy update.\n</system-update>".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::User {
                content: "hello".to_owned(),
                images: Vec::new(),
            },
        ]
    );
}

#[test]
fn should_escape_chronological_system_update_wrapper_delimiters() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Existing context."},
            {"role": "system", "content": "The text <system-update> & </system-update> must stay literal."}
        ],
        "stream": true
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the system-update delimiter request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(905), request)
            .expect("a later system message should preserve wrapper delimiters as literal text");

    assert_eq!(
        chat_command.messages,
        vec![ChatMessage::User {
            content: "Existing context.\n<system-update>\nThe text &lt;system-update&gt; &amp; &lt;/system-update&gt; must stay literal.\n</system-update>".to_owned(),
            images: Vec::new(),
        }]
    );
}

#[test]
fn should_translate_captured_opencode_reasoning_effort_into_the_thinking_budget() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "system", "content": "You generate concise conversation titles."},
            {"role": "user", "content": "Summarize this coding task."}
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 1024,
        "temperature": 0.5,
        "reasoning_effort": "low"
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the captured OpenCode title request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(901), request)
            .expect("the captured OpenCode effort level should translate into a budget");

    assert_eq!(
        chat_command,
        ChatGenerationCommand {
            request_id: RequestId::new(901),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![
                ChatMessage::System {
                    content: "You generate concise conversation titles.".to_owned(),
                },
                ChatMessage::User {
                    content: "Summarize this coding task.".to_owned(),
                    images: Vec::new(),
                },
            ],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Auto,
            settings: ChatGenerationSettings {
                max_output_tokens: 1024,
                temperature_thousandths: Some(500),
                top_p_thousandths: None,
                repetition_penalty_thousandths: None,
                seed: None,
                thinking_budget: Some(2048),
            },
            structured_generation: None,
        }
    );
}

#[test]
fn should_translate_opencode_large_output_budget_to_the_worker() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Inspect the repository and make all necessary edits."}
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 20000
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the OpenCode large-output request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(903), request)
            .expect("a 20,000-token output budget should reach model-serving admission");

    assert_eq!(chat_command.settings.max_output_tokens, 20_000);
}

#[test]
fn should_translate_installed_opencode_bash_tool_description_within_public_rest_limit() {
    let installed_opencode_bash_description_bytes = 4_672_usize;
    let opencode_bash_description = "x".repeat(installed_opencode_bash_description_bytes);
    let request_json = format!(
        r#"{{
            "model": "astronomical/fake-mixture-of-experts",
            "messages": [{{"role": "user", "content": "Run a smoke test."}}],
            "tools": [
                {{
                    "type": "function",
                    "function": {{
                        "name": "bash",
                        "description": {},
                        "parameters": {{"type": "object"}}
                    }}
                }}
            ],
            "tool_choice": "auto",
            "stream": true
        }}"#,
        serde_json::to_string(&opencode_bash_description)
            .expect("the representative OpenCode bash tool description should serialize")
    );
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(&request_json)
        .expect("the representative OpenCode tool request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(902), request)
            .expect("tool descriptions within the public REST limit should pass IPC validation");

    assert_eq!(chat_command.tools.len(), 1);
    assert_eq!(chat_command.tools[0].name, "bash");
    assert_eq!(
        chat_command.tools[0].description.as_deref(),
        Some(opencode_bash_description.as_str())
    );
}

#[test]
fn should_translate_the_current_opencode_tool_result_wire_shape_without_rest_dtos_crossing_ipc() {
    let request_json = r#"
    {
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "List Rust source files."},
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "I should inspect the source tree.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "glob", "arguments": "{\"pattern\":\"src/**/*.rs\"}"}
                    }
                ]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "src/lib.rs"},
            {"role": "user", "content": "Summarize the source files."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "List matching files.",
                    "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}
                }
            }
        ],
        "tool_choice": "auto",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 7
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the representative current OpenCode request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(900), request)
            .expect("the validated REST request should translate to independent IPC DTOs");

    assert_eq!(
        chat_command,
        ChatGenerationCommand {
            request_id: RequestId::new(900),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![
                ChatMessage::System {
                    content: "You are a coding assistant.".to_owned(),
                },
                ChatMessage::User {
                    content: "List Rust source files.".to_owned(),
                    images: Vec::new(),
                },
                ChatMessage::Assistant {
                    content: None,
                    reasoning_content: Some("I should inspect the source tree.".to_owned()),
                    tool_calls: vec![ChatAssistantToolCall {
                        id: "call_1".to_owned(),
                        function: ChatAssistantToolFunction {
                            name: "glob".to_owned(),
                            arguments_json: r#"{"pattern":"src/**/*.rs"}"#.to_owned(),
                        },
                    }],
                },
                ChatMessage::Tool {
                    tool_call_id: "call_1".to_owned(),
                    content: "src/lib.rs".to_owned(),
                },
                ChatMessage::User {
                    content: "Summarize the source files.".to_owned(),
                    images: Vec::new(),
                },
            ],
            tools: vec![ChatToolDefinition {
                name: "glob".to_owned(),
                description: Some("List matching files.".to_owned()),
                parameters_json: r#"{"properties":{"pattern":{"type":"string"}},"type":"object"}"#
                    .to_owned(),
            }],
            tool_choice: ChatToolChoice::Auto,
            settings: ChatGenerationSettings {
                max_output_tokens: 512,
                temperature_thousandths: Some(600),
                top_p_thousandths: Some(950),
                repetition_penalty_thousandths: None,
                seed: Some(7),
                thinking_budget: None,
            },
            structured_generation: None,
        }
    );
}

#[test]
fn should_inject_a_json_instruction_when_response_format_is_json_object() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "response_format": {"type": "json_object"}
        }"#,
    )
    .expect("json_object chat request should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(399), request)
            .expect("json_object should translate with a JSON instruction");

    match chat_command.messages.as_slice() {
        [ChatMessage::System { content }, ChatMessage::User { .. }] => {
            assert!(content.contains("JSON"));
        }
        other_messages => panic!("expected a leading system instruction, got {other_messages:?}"),
    }
}

#[test]
fn should_translate_structured_outputs_choice_into_an_ipc_mask() {
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(
        r#"{
            "model": "mlx-community/Qwen3.5-2B-4bit",
            "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
            "structured_outputs": {"choice": ["Juliet", "Romeo"]}
        }"#,
    )
    .expect("choice extra body should deserialize");

    let chat_command =
        crate::translate_openai_chat_completion_request(RequestId::new(410), request)
            .expect("choice extra body should translate to an IPC mask");

    assert_eq!(
        chat_command.structured_generation,
        Some(StructuredGenerationConstraint::Choice {
            choices: vec!["Juliet".to_owned(), "Romeo".to_owned()],
        })
    );
}

/// A 1x1 red PNG, the smallest image the decoder accepts as real.
const RED_PIXEL_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";

#[test]
fn should_follow_tool_messages_with_the_images_they_returned() {
    let request = serde_json::from_value::<OpenAiChatCompletionRequest>(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Capture the stage."},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": [
                {"type": "text", "text": "captured"},
                {"type": "image_url", "image_url": {"url": RED_PIXEL_DATA_URI}}
            ]},
            {"role": "tool", "tool_call_id": "call_2", "content": "Exit, pursued by a bear."},
            {"role": "user", "content": "Describe it."}
        ]
    }))
    .expect("the chat request should deserialize");

    let chat_messages =
        crate::translate_openai_chat_completion_request(RequestId::new(911), request)
            .expect("the chat request should translate")
            .messages;

    assert!(matches!(
        &chat_messages[2..],
        [
            ChatMessage::Tool { tool_call_id: first_call, content: first_content },
            ChatMessage::Tool { tool_call_id: second_call, .. },
            ChatMessage::User { content: image_message_text, images },
            ChatMessage::User { content: question, images: question_images },
        ] if first_call == "call_1"
            && first_content == "captured"
            && second_call == "call_2"
            && image_message_text == "Images returned by the tool calls above."
            && images.len() == 1
            && question == "Describe it."
            && question_images.is_empty()
    ));
}

/// One assistant tool call to the `screenshot` tool.
fn screenshot_tool_call(call_id: &str) -> serde_json::Value {
    serde_json::json!({"id": call_id, "type": "function", "function": {"name": "screenshot", "arguments": "{}"}})
}

/// A tool message returning one screenshot for `call_id`.
fn screenshot_result(call_id: &str) -> serde_json::Value {
    serde_json::json!({"role": "tool", "tool_call_id": call_id, "content": [
        {"type": "image_url", "image_url": {"url": RED_PIXEL_DATA_URI}}
    ]})
}

#[test]
fn should_place_each_tool_runs_images_after_that_run() {
    let request = serde_json::from_value::<OpenAiChatCompletionRequest>(serde_json::json!({
        "model": "astronomical/fake-mixture-of-experts",
        "messages": [
            {"role": "user", "content": "Capture the stage twice."},
            {"role": "assistant", "content": null, "tool_calls": [screenshot_tool_call("call_1")]},
            screenshot_result("call_1"),
            {"role": "assistant", "content": null, "tool_calls": [screenshot_tool_call("call_2")]},
            screenshot_result("call_2"),
            {"role": "user", "content": "Compare them."}
        ]
    }))
    .expect("the chat request should deserialize");

    let chat_messages =
        crate::translate_openai_chat_completion_request(RequestId::new(912), request)
            .expect("the chat request should translate")
            .messages;

    let image_message_positions: Vec<usize> = chat_messages
        .iter()
        .enumerate()
        .filter(|(_, chat_message)| {
            matches!(chat_message, ChatMessage::User { images, .. } if !images.is_empty())
        })
        .map(|(message_position, _)| message_position)
        .collect();
    assert_eq!(
        image_message_positions,
        [3, 6],
        "each run's image follows its own tool message, before the next assistant turn"
    );
}
