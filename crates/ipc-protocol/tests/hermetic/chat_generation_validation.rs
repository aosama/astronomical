use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatGenerationValidationError, ChatMessage, ChatToolChoice,
    ChatToolDefinition, RequestId,
};

#[test]
fn should_reject_an_empty_model_id_before_worker_preprocessing() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: String::new(),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::EmptyModelId)
    );
}

#[test]
fn should_accept_a_large_model_id_when_the_ipc_frame_fits() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "m".repeat(257),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(950),
            seed: Some(7),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };

    chat_generation_command
        .validate()
        .expect("the IPC frame-size limit should bound model IDs, not a tiny field cap");
}

#[test]
fn should_reject_a_temperature_above_the_supported_sampling_range() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: Some(2_001),
            top_p_thousandths: Some(950),
            seed: Some(7),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };

    assert_eq!(
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::TemperatureOutOfRange {
            actual_temperature_thousandths: 2_001,
            maximum_temperature_thousandths: 2_000,
        })
    );
}

#[test]
fn should_reject_a_top_p_above_the_supported_sampling_range() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(1_001),
            seed: Some(7),
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };

    assert_eq!(
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::TopPOutOfRange {
            actual_top_p_thousandths: 1_001,
            maximum_top_p_thousandths: 1_000,
        })
    );
}

#[test]
fn should_reject_malformed_assistant_tool_call_arguments() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![ChatAssistantToolCall {
                id: "call_1".to_owned(),
                function: ChatAssistantToolFunction {
                    name: "glob".to_owned(),
                    arguments_json: "not JSON".to_owned(),
                },
            }],
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(
            ChatGenerationValidationError::InvalidAssistantToolCallArguments {
                tool_call_id: "call_1".to_owned(),
            }
        )
    );
}

#[test]
fn should_reject_an_assistant_tool_call_argument_json_value_that_is_not_an_object() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: vec![ChatAssistantToolCall {
                id: "call_1".to_owned(),
                function: ChatAssistantToolFunction {
                    name: "glob".to_owned(),
                    arguments_json: "[]".to_owned(),
                },
            }],
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(
            ChatGenerationValidationError::AssistantToolCallArgumentsMustBeObject {
                tool_call_id: "call_1".to_owned(),
            }
        )
    );
}

#[test]
fn should_reject_an_empty_declared_tool_name_before_prompt_rendering() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(60),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }],
        tools: vec![ChatToolDefinition {
            name: String::new(),
            description: None,
            parameters_json: "{}".to_owned(),
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
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::EmptyToolDefinitionName)
    );
}

#[test]
fn should_reject_a_duplicate_assistant_tool_call_id() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(61),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "Inspect the repository.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![
                    ChatAssistantToolCall {
                        id: "call_1".to_owned(),
                        function: ChatAssistantToolFunction {
                            name: "glob".to_owned(),
                            arguments_json: "{}".to_owned(),
                        },
                    },
                    ChatAssistantToolCall {
                        id: "call_1".to_owned(),
                        function: ChatAssistantToolFunction {
                            name: "glob".to_owned(),
                            arguments_json: "{}".to_owned(),
                        },
                    },
                ],
            },
        ],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(
            ChatGenerationValidationError::DuplicateAssistantToolCallId {
                tool_call_id: "call_1".to_owned(),
            }
        )
    );
}

#[test]
fn should_accept_a_reused_tool_call_id_after_its_previous_result() {
    let tool_call = || ChatAssistantToolCall {
        id: "call_chatcmpl-5_0".to_owned(),
        function: ChatAssistantToolFunction {
            name: "read".to_owned(),
            arguments_json: r#"{"filePath":"README.md"}"#.to_owned(),
        },
    };
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(63),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "Read the file.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![tool_call()],
            },
            ChatMessage::Tool {
                tool_call_id: "call_chatcmpl-5_0".to_owned(),
                content: "first result".to_owned(),
            },
            ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![tool_call()],
            },
            ChatMessage::Tool {
                tool_call_id: "call_chatcmpl-5_0".to_owned(),
                content: "second result".to_owned(),
            },
            ChatMessage::User {
                content: "Summarize both results.".to_owned(),
                images: Vec::new(),
            },
        ],
        tools: vec![ChatToolDefinition {
            name: "read".to_owned(),
            description: None,
            parameters_json: r#"{"type":"object","properties":{"filePath":{"type":"string"}}}"#
                .to_owned(),
        }],
        tool_choice: ChatToolChoice::Auto,
        settings: ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
        qwen_thinking_channel_seed: None,
    };

    assert_eq!(chat_generation_command.validate(), Ok(()));
}

#[test]
fn should_reject_a_system_message_after_conversation_history_begins() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(62),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "Inspect the repository.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::System {
                content: "This placement is invalid.".to_owned(),
            },
        ],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::SystemMessageMustBeFirst { message_index: 1 })
    );
}

#[test]
fn should_reject_a_tool_result_without_a_prior_assistant_tool_call() {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(63),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "Inspect the repository.".to_owned(),
                images: Vec::new(),
            },
            ChatMessage::Tool {
                tool_call_id: "call_missing".to_owned(),
                content: "No matching call exists.".to_owned(),
            },
        ],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
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
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::UnknownToolResultId {
            tool_call_id: "call_missing".to_owned(),
        })
    );
}
