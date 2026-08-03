use super::*;

#[test]
fn should_deserialize_a_standard_streaming_tool_use_request() {
    let request_json = r#"
    {
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "List Rust source files."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "List matching files.",
                    "parameters": {
                        "type": "object",
                        "properties": {"pattern": {"type": "string"}},
                        "required": ["pattern"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    "#;

    let chat_completion_request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the standard OpenAI-compatible tool-use request should deserialize");

    chat_completion_request
        .validate()
        .expect("the bounded standard tool-use request should validate");
    assert_eq!(
        chat_completion_request.model(),
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit"
    );
    assert_eq!(chat_completion_request.messages().len(), 2);
    assert_eq!(chat_completion_request.tools().len(), 1);
    assert!(chat_completion_request.stream());
    assert!(chat_completion_request.includes_usage_in_stream());
}

#[test]
fn should_expose_validated_request_parts_without_leaking_rest_dtos_into_ipc() {
    let request_json = r#"
    {
        "model": "mlx-community/Ornith-1.0-35B-OptiQ-4bit",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Inspect "}, {"type": "text", "text": "the repository."}]}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "List matching files.",
                    "parameters": {"type": "object"}
                }
            }
        ],
        "tool_choice": "none",
        "max_completion_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": 7,
        "stream": true,
        "stream_options": {"include_usage": true}
    }
    "#;
    let request = serde_json::from_str::<OpenAiChatCompletionRequest>(request_json)
        .expect("the request should deserialize");

    let request_parts = request
        .into_parts()
        .expect("the validated request should expose conversion parts");

    assert_eq!(
        request_parts.model,
        "mlx-community/Ornith-1.0-35B-OptiQ-4bit"
    );
    assert_eq!(request_parts.maximum_output_tokens, 512);
    assert_eq!(request_parts.tool_choice, OpenAiToolChoiceMode::None);
    assert_eq!(request_parts.temperature, Some(0.6));
    assert_eq!(request_parts.top_p, Some(0.95));
    assert_eq!(request_parts.seed, Some(7));
    assert!(request_parts.stream);
    assert!(request_parts.includes_usage_in_stream);
    assert_eq!(request_parts.tools[0].name, "glob");
    assert_eq!(
        request_parts.tools[0].parameters_json,
        r#"{"type":"object"}"#
    );
    assert!(matches!(
        request_parts.messages.as_slice(),
        [OpenAiChatMessageParts::User { content, .. }] if content == "Inspect the repository."
    ));
}
