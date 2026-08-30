use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationFailureReason, ChatGenerationSettings, ChatMessage, ChatModelCapabilities,
    ChatToolChoice, ChatToolDefinition, RequestId,
};

#[test]
fn should_serialize_one_structured_chat_generation_command_without_rest_types() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(44),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::System {
                content: "You are a coding assistant.".to_owned(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: Some("I should inspect the repository.".to_owned()),
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
            parameters_json: r#"{"type":"object","properties":{"pattern":{"type":"string"}}}"#
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
    };

    assert_eq!(
        serde_json::to_string(&chat_generation_command)
            .expect("the structured chat payload should serialize"),
        r#"{"request_id":44,"model":"astronomical/fake-mixture-of-experts","messages":[{"role":"system","content":"You are a coding assistant."},{"role":"assistant","content":null,"reasoning_content":"I should inspect the repository.","tool_calls":[{"id":"call_1","function":{"name":"glob","arguments_json":"{\"pattern\":\"src/**/*.rs\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"src/lib.rs"},{"role":"user","content":"Summarize the source files.","images":[]}],"tools":[{"name":"glob","description":"List matching files.","parameters_json":"{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}}}"}],"tool_choice":{"kind":"auto"},"settings":{"max_output_tokens":512,"temperature_thousandths":600,"top_p_thousandths":950,"seed":7,"thinking_budget":null}}"#
    );
}

#[test]
fn should_round_trip_a_typed_context_length_exceeded_failure() {
    let failure_reason = ChatGenerationFailureReason::ContextLengthExceeded {
        actual_total_context_tokens: 262_145,
        maximum_context_tokens: 262_144,
    };

    let serialized_failure =
        serde_json::to_string(&failure_reason).expect("the context failure should serialize");
    let deserialized_failure =
        serde_json::from_str::<ChatGenerationFailureReason>(&serialized_failure)
            .expect("the context failure should deserialize");

    assert_eq!(deserialized_failure, failure_reason);
    assert!(serialized_failure.contains("context_length_exceeded"));
    assert!(serialized_failure.contains("262145"));
}

#[test]
fn should_round_trip_ready_model_capabilities_with_vision_support() {
    let model_capabilities = ChatModelCapabilities {
        supports_reasoning: true,
        supports_tool_calls: true,
        has_vision: true,
        max_input_tokens: 241_664,
        max_output_tokens: 20_480,
        context_window: 262_144,
    };

    let serialized_model_capabilities = serde_json::to_string(&model_capabilities)
        .expect("ready model capabilities should serialize");
    let deserialized_model_capabilities =
        serde_json::from_str::<ChatModelCapabilities>(&serialized_model_capabilities)
            .expect("ready model capabilities should deserialize");

    assert_eq!(deserialized_model_capabilities, model_capabilities);
    assert!(serialized_model_capabilities.contains(r#""has_vision":true"#));
}

#[test]
fn should_round_trip_one_chat_command_with_a_user_message_image() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(99),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::User {
            content: "What is in this picture?".to_owned(),
            images: vec![astronomical_ipc_protocol::ChatImageInput {
                mime_type: "image/png".to_owned(),
                decoded_bytes: vec![0x89, 0x50, 0x4E, 0x47],
            }],
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };

    let serialized_json = serde_json::to_string(&chat_generation_command)
        .expect("the chat command with one image should serialize");
    assert!(serialized_json.contains(r#""decoded_bytes":"iVBORw==""#));
    assert!(!serialized_json.contains(r#""decoded_bytes":["#));
    let deserialized_command: ChatGenerationCommand = serde_json::from_str(&serialized_json)
        .expect("the chat command with one image should round-trip");
    assert_eq!(deserialized_command, chat_generation_command);
    let user_message = match deserialized_command.messages.as_slice() {
        [ChatMessage::User { content, images }] => {
            assert_eq!(content, "What is in this picture?");
            images
        }
        other => panic!("expected one user message with images, got {other:?}"),
    };
    assert_eq!(user_message.len(), 1);
    assert_eq!(user_message[0].mime_type, "image/png");
    assert_eq!(user_message[0].decoded_bytes, vec![0x89, 0x50, 0x4E, 0x47]);
}

#[test]
fn should_round_trip_a_qwen_thinking_channel_seed() {
    let chat_generation_command = chat_generation_command_with_seed(
        "Two households, both alike in dignity, in Romeo and Juliet.".to_owned(),
    );

    let serialized_json = serde_json::to_string(&chat_generation_command)
        .expect("a seeded chat command should serialize");
    let deserialized_command: ChatGenerationCommand =
        serde_json::from_str(&serialized_json).expect("a seeded chat command should round-trip");

    assert_eq!(deserialized_command, chat_generation_command);
    assert!(serialized_json.contains("qwen_thinking_channel_seed"));
    assert!(
        serialized_json.contains("Two households, both alike in dignity, in Romeo and Juliet.")
    );
}

#[test]
fn should_reject_a_qwen_thinking_channel_seed_that_exceeds_its_bounded_contract() {
    let chat_generation_command = chat_generation_command_with_seed(
        "R".repeat(astronomical_ipc_protocol::MAX_QWEN_THINKING_CHANNEL_SEED_BYTES + 1),
    );

    assert!(matches!(
        chat_generation_command.validate(),
        Err(astronomical_ipc_protocol::ChatGenerationValidationError::QwenThinkingChannelSeedTooLarge { .. })
    ));
}

fn chat_generation_command_with_seed(qwen_thinking_channel_seed: String) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(105),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Who is Romeo?".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 16,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: Some(qwen_thinking_channel_seed),
    }
}
