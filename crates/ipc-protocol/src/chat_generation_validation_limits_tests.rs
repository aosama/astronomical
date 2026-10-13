use astronomical_test_harness::offline_denied_test;

use crate::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatGenerationValidationError, ChatMessage, ChatToolChoice,
    ChatToolDefinition, MAX_IPC_FRAME_BYTES, RequestId, WorkerCommand,
};

offline_denied_test!(
    should_reject_a_second_tool_result_for_the_same_assistant_tool_call,
    {
        let assistant_tool_call = ChatAssistantToolCall {
            id: "call_1".to_owned(),
            function: ChatAssistantToolFunction {
                name: "glob".to_owned(),
                arguments_json: "{}".to_owned(),
            },
        };
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(64),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![
                ChatMessage::Assistant {
                    content: None,
                    reasoning_content: None,
                    tool_calls: vec![assistant_tool_call],
                },
                ChatMessage::Tool {
                    tool_call_id: "call_1".to_owned(),
                    content: "first result".to_owned(),
                },
                ChatMessage::Tool {
                    tool_call_id: "call_1".to_owned(),
                    content: "second result".to_owned(),
                },
            ],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        assert_eq!(
            chat_generation_command.validate(),
            Err(ChatGenerationValidationError::DuplicateToolResultId {
                tool_call_id: "call_1".to_owned(),
            })
        );
    }
);

offline_denied_test!(should_reject_duplicate_declared_tool_names, {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(65),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: support::user_messages(),
        tools: vec![
            ChatToolDefinition {
                name: "glob".to_owned(),
                description: None,
                parameters_json: "{}".to_owned(),
            },
            ChatToolDefinition {
                name: "glob".to_owned(),
                description: None,
                parameters_json: "{}".to_owned(),
            },
        ],
        tool_choice: ChatToolChoice::Auto,
        settings: support::standard_settings(),
        structured_generation: None,
    };

    assert_eq!(
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::DuplicateToolDefinitionName {
            function_name: "glob".to_owned(),
        })
    );
});

offline_denied_test!(
    should_reject_a_forced_tool_choice_for_an_undeclared_function,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(66),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Function {
                name: "glob".to_owned(),
            },
            settings: support::standard_settings(),
            structured_generation: None,
        };

        assert_eq!(
            chat_generation_command.validate(),
            Err(
                ChatGenerationValidationError::ToolChoiceNamesUnknownFunction {
                    function_name: "glob".to_owned(),
                }
            )
        );
    }
);

offline_denied_test!(
    should_reject_required_tool_choice_before_prompt_rendering,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(79),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Required,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        assert_eq!(
            chat_generation_command.validate(),
            Err(ChatGenerationValidationError::UnsupportedToolChoice { mode: "required" })
        );
    }
);

offline_denied_test!(
    should_reject_a_declared_forced_tool_choice_before_prompt_rendering,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(80),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: vec![ChatToolDefinition {
                name: "glob".to_owned(),
                description: None,
                parameters_json: "{}".to_owned(),
            }],
            tool_choice: ChatToolChoice::Function {
                name: "glob".to_owned(),
            },
            settings: support::standard_settings(),
            structured_generation: None,
        };

        assert_eq!(
            chat_generation_command.validate(),
            Err(ChatGenerationValidationError::UnsupportedToolChoice { mode: "function" })
        );
    }
);

offline_denied_test!(should_reject_a_tool_schema_deeper_than_the_worker_limit, {
    let deeply_nested_schema = (0..33).fold("{}".to_owned(), |nested_schema, _| {
        format!(r#"{{"items":{nested_schema}}}"#)
    });
    let chat_generation_command = support::command_with_tool_schema(67, deeply_nested_schema);

    assert!(matches!(
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::ToolSchemaNestingTooDeep {
            function_name,
            maximum_schema_nesting_depth: 32,
            ..
        }) if function_name == "glob"
    ));
});

offline_denied_test!(should_accept_a_large_tool_schema_when_the_ipc_frame_fits, {
    let oversized_schema = format!(r#"{{"description":"{}"}}"#, "x".repeat(32 * 1024));
    let chat_generation_command = support::command_with_tool_schema(68, oversized_schema);

    chat_generation_command
        .validate()
        .expect("IPC frame bytes should bound tool schemas, not a per-schema byte cap");
});

offline_denied_test!(
    should_accept_large_aggregate_tool_schemas_when_the_ipc_frame_fits,
    {
        let schema_with_large_description =
            format!(r#"{{"description":"{}"}}"#, "x".repeat(25 * 1024));
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(69),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: vec![
                ChatToolDefinition {
                    name: "glob".to_owned(),
                    description: None,
                    parameters_json: schema_with_large_description.clone(),
                },
                ChatToolDefinition {
                    name: "grep".to_owned(),
                    description: None,
                    parameters_json: schema_with_large_description,
                },
            ],
            tool_choice: ChatToolChoice::Auto,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command.validate().expect(
            "IPC frame bytes should bound aggregate tool schemas, not a total schema byte cap",
        );
    }
);

offline_denied_test!(
    should_reject_an_empty_chat_history_before_worker_preprocessing,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(70),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        assert_eq!(
            chat_generation_command.validate(),
            Err(ChatGenerationValidationError::EmptyMessages)
        );
    }
);

offline_denied_test!(
    should_accept_large_chat_history_without_worker_message_count_cap,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(81),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: (0..250)
                .map(|message_number| ChatMessage::User {
                    content: format!("short turn {message_number}"),
                    images: Vec::new(),
                })
                .collect(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command.validate().expect(
            "IPC frame bytes and model context should bound long histories, not message count",
        );
    }
);

offline_denied_test!(
    should_accept_many_small_tool_definitions_without_worker_tool_count_cap,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(82),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: (0..250)
                .map(|tool_number| ChatToolDefinition {
                    name: format!("tool_{tool_number}"),
                    description: None,
                    parameters_json: "{}".to_owned(),
                })
                .collect(),
            tool_choice: ChatToolChoice::Auto,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command
            .validate()
            .expect("IPC frame bytes and schema bytes should bound tools, not tool count");
    }
);

offline_denied_test!(should_reject_a_zero_structured_chat_output_token_budget, {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(71),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: support::user_messages(),
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: 0,
            ..support::standard_settings()
        },
        structured_generation: None,
    };

    assert_eq!(
        chat_generation_command.validate(),
        Err(ChatGenerationValidationError::OutputTokenCountOutOfRange {
            actual_output_tokens: 0,
            maximum_output_tokens: u16::MAX,
        })
    );
});

offline_denied_test!(
    should_accept_large_structured_chat_output_budget_for_model_context_admission,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(72),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: ChatGenerationSettings {
                max_output_tokens: 20_000,
                ..support::standard_settings()
            },
            structured_generation: None,
        };

        assert_eq!(chat_generation_command.validate(), Ok(()));
    }
);

offline_denied_test!(
    should_accept_one_chat_message_larger_than_the_old_message_byte_limit_when_the_ipc_frame_fits,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(74),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![ChatMessage::User {
                content: "x".repeat(48 * 1024 + 1),
                images: Vec::new(),
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };
        let serialized_command_bytes =
            serde_json::to_vec(&WorkerCommand::Generate(chat_generation_command.clone()))
                .expect("the typed command should serialize")
                .len();
        assert!(serialized_command_bytes < MAX_IPC_FRAME_BYTES);

        assert_eq!(chat_generation_command.validate(), Ok(()));
    }
);

offline_denied_test!(should_accept_aggregate_chat_messages_larger_than_the_old_message_byte_limit_when_the_ipc_frame_fits, {
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(75),
        model: "astronomical/fake-mixture-of-experts".to_owned(),
        messages: vec![
            ChatMessage::User {
                content: "a".repeat(16 * 1024),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: Some("b".repeat(16 * 1024)),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
            ChatMessage::User {
                content: "c".repeat(16 * 1024),
                images: Vec::new(),
            },
            ChatMessage::Assistant {
                content: Some("d".to_owned()),
                reasoning_content: None,
                tool_calls: Vec::new(),
            },
        ],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: support::standard_settings(),
        structured_generation: None,
    };
    let serialized_command_bytes =
        serde_json::to_vec(&WorkerCommand::Generate(chat_generation_command.clone()))
            .expect("the typed command should serialize")
            .len();
    assert!(serialized_command_bytes < MAX_IPC_FRAME_BYTES);

    assert_eq!(chat_generation_command.validate(), Ok(()));
});

offline_denied_test!(
    should_accept_a_semantically_valid_large_chat_command_that_fits_one_ipc_frame,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(76),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![
                ChatMessage::User {
                    content: "a".repeat(16 * 1024),
                    images: Vec::new(),
                },
                ChatMessage::Assistant {
                    content: Some("b".repeat(16 * 1024)),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                },
                ChatMessage::User {
                    content: "c".repeat(16 * 1024),
                    images: Vec::new(),
                },
            ],
            tools: vec![ChatToolDefinition {
                name: "glob".to_owned(),
                description: None,
                parameters_json: format!(r#"{{"description":"{}"}}"#, "x".repeat(16 * 1024)),
            }],
            tool_choice: ChatToolChoice::Auto,
            settings: support::standard_settings(),
            structured_generation: None,
        };
        let serialized_bytes =
            serde_json::to_vec(&WorkerCommand::Generate(chat_generation_command.clone()))
                .expect("the bounded typed command should serialize")
                .len();
        assert!(serialized_bytes > 64 * 1024);
        assert!(serialized_bytes <= MAX_IPC_FRAME_BYTES);

        assert_eq!(chat_generation_command.validate(), Ok(()));
    }
);

offline_denied_test!(
    should_accept_a_large_tool_description_when_the_ipc_frame_fits,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(77),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: support::user_messages(),
            tools: vec![ChatToolDefinition {
                name: "glob".to_owned(),
                description: Some("x".repeat(8 * 1024 + 1)),
                parameters_json: "{}".to_owned(),
            }],
            tool_choice: ChatToolChoice::Auto,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command.validate().expect(
            "IPC frame bytes should bound tool descriptions, not a per-description byte cap",
        );
    }
);

offline_denied_test!(
    should_accept_a_large_assistant_tool_call_id_when_the_ipc_frame_fits,
    {
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(77),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![ChatAssistantToolCall {
                    id: "x".repeat(128 + 1),
                    function: ChatAssistantToolFunction {
                        name: "glob".to_owned(),
                        arguments_json: "{}".to_owned(),
                    },
                }],
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command
            .validate()
            .expect("IPC frame bytes should bound assistant tool-call IDs, not a tiny field cap");
    }
);

offline_denied_test!(
    should_accept_large_assistant_tool_call_arguments_when_the_ipc_frame_fits,
    {
        let large_arguments_json = format!(r#"{{"payload":"{}"}}"#, "x".repeat(16_665));
        let chat_generation_command = ChatGenerationCommand {
            request_id: RequestId::new(83),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: vec![ChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                tool_calls: vec![ChatAssistantToolCall {
                    id: "call_5y44arzw".to_owned(),
                    function: ChatAssistantToolFunction {
                        name: "read".to_owned(),
                        arguments_json: large_arguments_json,
                    },
                }],
            }],
            tools: Vec::new(),
            tool_choice: ChatToolChoice::None,
            settings: support::standard_settings(),
            structured_generation: None,
        };

        chat_generation_command.validate().expect(
            "IPC frame bytes should bound assistant tool-call arguments, not the old 8192-byte cap",
        );
    }
);

mod support {

    use crate::{
        ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice,
        ChatToolDefinition, RequestId,
    };

    pub(super) fn command_with_tool_schema(
        request_id: u64,
        parameters_json: String,
    ) -> ChatGenerationCommand {
        ChatGenerationCommand {
            request_id: RequestId::new(request_id),
            model: "astronomical/fake-mixture-of-experts".to_owned(),
            messages: user_messages(),
            tools: vec![ChatToolDefinition {
                name: "glob".to_owned(),
                description: None,
                parameters_json,
            }],
            tool_choice: ChatToolChoice::Auto,
            settings: standard_settings(),
            structured_generation: None,
        }
    }
    pub(super) fn standard_settings() -> ChatGenerationSettings {
        ChatGenerationSettings {
            max_output_tokens: 512,
            temperature_thousandths: Some(600),
            top_p_thousandths: Some(950),
            seed: Some(7),
            thinking_budget: None,
        }
    }
    pub(super) fn user_messages() -> Vec<ChatMessage> {
        vec![ChatMessage::User {
            content: "Inspect the repository.".to_owned(),
            images: Vec::new(),
        }]
    }
}
