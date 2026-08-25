use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_rest_contract::OpenAiChatCompletionRequest;
use astronomical_supervisor::translate_openai_chat_completion_request;

#[test]
fn should_lower_a_later_system_message_to_a_chronological_user_update() {
    let request_json = r#"
    {
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
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

    let chat_command = translate_openai_chat_completion_request(RequestId::new(904), request)
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
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        "messages": [
            {"role": "user", "content": "Existing context."},
            {"role": "system", "content": "The text <system-update> & </system-update> must stay literal."}
        ],
        "stream": true
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the system-update delimiter request should deserialize");

    let chat_command = translate_openai_chat_completion_request(RequestId::new(905), request)
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
fn should_ignore_captured_opencode_reasoning_effort_when_translating_to_ipc() {
    let request_json = r#"
    {
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
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

    let chat_command = translate_openai_chat_completion_request(RequestId::new(901), request)
        .expect("reasoning_effort should be accepted as an ignored REST-only option");

    assert_eq!(
        chat_command,
        ChatGenerationCommand {
            request_id: RequestId::new(901),
            model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
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
                seed: None,
                thinking_budget: None,
            },
            qwen_thinking_channel_seed: None,
        }
    );
}

#[test]
fn should_translate_opencode_large_output_budget_to_the_worker() {
    let request_json = r#"
    {
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
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

    let chat_command = translate_openai_chat_completion_request(RequestId::new(903), request)
        .expect("a 20,000-token output budget should reach model-serving admission");

    assert_eq!(chat_command.settings.max_output_tokens, 20_000);
}

#[test]
fn should_translate_installed_opencode_bash_tool_description_within_public_rest_limit() {
    let installed_opencode_bash_description_bytes = 4_672_usize;
    let opencode_bash_description = "x".repeat(installed_opencode_bash_description_bytes);
    let request_json = format!(
        r#"{{
            "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
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

    let chat_command = translate_openai_chat_completion_request(RequestId::new(902), request)
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
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
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

    let chat_command = translate_openai_chat_completion_request(RequestId::new(900), request)
        .expect("the validated REST request should translate to independent IPC DTOs");

    assert_eq!(
        chat_command,
        ChatGenerationCommand {
            request_id: RequestId::new(900),
            model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
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
                seed: Some(7),
                thinking_budget: None,
            },
            qwen_thinking_channel_seed: None,
        }
    );
}
