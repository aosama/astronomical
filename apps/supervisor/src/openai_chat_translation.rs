use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatGenerationValidationError, ChatImageInput, ChatMessage,
    ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_rest_contract::{
    OpenAiChatCompletionRequest, OpenAiChatCompletionRequestParts,
    OpenAiChatCompletionValidationError, OpenAiChatMessageParts, OpenAiToolChoiceMode,
    OpenAiToolDefinitionParts,
};
use thiserror::Error;

const CHRONOLOGICAL_SYSTEM_UPDATE_OPENING_TAG: &str = "<system-update>\n";
const CHRONOLOGICAL_SYSTEM_UPDATE_CLOSING_TAG: &str = "\n</system-update>";

/// Translates one validated public OpenAI Chat Completions request into independent IPC data.
pub fn translate_openai_chat_completion_request(
    request_id: RequestId,
    request: OpenAiChatCompletionRequest,
) -> Result<ChatGenerationCommand, OpenAiChatTranslationError> {
    let request_parts = request
        .into_parts()
        .map_err(OpenAiChatTranslationError::PublicValidation)?;
    translate_openai_chat_completion_request_parts(request_id, request_parts)
}

pub(crate) fn translate_openai_chat_completion_request_parts(
    request_id: RequestId,
    request_parts: OpenAiChatCompletionRequestParts,
) -> Result<ChatGenerationCommand, OpenAiChatTranslationError> {
    let OpenAiChatCompletionRequestParts {
        model,
        messages,
        tools,
        tool_choice,
        maximum_output_tokens,
        requested_maximum_output_tokens: _,
        temperature,
        top_p,
        seed,
        thinking_budget,
        stream: _,
        includes_usage_in_stream: _,
    } = request_parts;
    let maximum_output_tokens = u16::try_from(maximum_output_tokens).map_err(|_| {
        OpenAiChatTranslationError::OutputTokenCountTooLarge {
            actual_output_tokens: maximum_output_tokens,
        }
    })?;
    let thinking_budget = thinking_budget
        .map(u16::try_from)
        .transpose()
        .map_err(|_| OpenAiChatTranslationError::ThinkingBudgetTooLarge)?;
    let chat_generation_command = ChatGenerationCommand {
        request_id,
        model,
        messages: translate_messages(messages),
        tools: translate_tools(tools),
        tool_choice: translate_tool_choice(tool_choice)?,
        settings: ChatGenerationSettings {
            max_output_tokens: maximum_output_tokens,
            temperature_thousandths: translate_thousandths(temperature, "temperature")?,
            top_p_thousandths: translate_thousandths(top_p, "top_p")?,
            seed,
            thinking_budget,
        },
    };
    chat_generation_command
        .validate()
        .map_err(OpenAiChatTranslationError::IpcValidation)?;
    Ok(chat_generation_command)
}

fn translate_messages(messages: Vec<OpenAiChatMessageParts>) -> Vec<ChatMessage> {
    let mut translated_chat_messages = Vec::with_capacity(messages.len());
    for openai_chat_message in messages {
        match openai_chat_message {
            OpenAiChatMessageParts::System {
                content: system_message_content,
            } => {
                if translated_chat_messages.is_empty() {
                    translated_chat_messages.push(ChatMessage::System {
                        content: system_message_content,
                    });
                } else {
                    append_chronological_system_update(
                        &mut translated_chat_messages,
                        &system_message_content,
                    );
                }
            }
            OpenAiChatMessageParts::User {
                content: user_message_content,
                images: user_message_images,
            } => translated_chat_messages.push(ChatMessage::User {
                content: user_message_content,
                images: translate_images(user_message_images),
            }),
            OpenAiChatMessageParts::Assistant {
                content: assistant_message_content,
                reasoning_content: assistant_reasoning_content,
                tool_calls: assistant_tool_calls,
            } => translated_chat_messages.push(ChatMessage::Assistant {
                content: assistant_message_content,
                reasoning_content: assistant_reasoning_content,
                tool_calls: assistant_tool_calls
                    .into_iter()
                    .map(|assistant_tool_call| ChatAssistantToolCall {
                        id: assistant_tool_call.id,
                        function: ChatAssistantToolFunction {
                            name: assistant_tool_call.name,
                            arguments_json: assistant_tool_call.arguments_json,
                        },
                    })
                    .collect(),
            }),
            OpenAiChatMessageParts::Tool {
                tool_call_id,
                content: tool_result_content,
            } => translated_chat_messages.push(ChatMessage::Tool {
                tool_call_id,
                content: tool_result_content,
            }),
        }
    }
    translated_chat_messages
}

fn append_chronological_system_update(
    translated_chat_messages: &mut Vec<ChatMessage>,
    system_update_content: &str,
) {
    if let Some(ChatMessage::User {
        content: prior_user_message_content,
        ..
    }) = translated_chat_messages.last_mut()
    {
        prior_user_message_content.push('\n');
        append_escaped_chronological_system_update(
            prior_user_message_content,
            system_update_content,
        );
        return;
    }

    let mut chronological_user_update_content = String::with_capacity(system_update_content.len());
    append_escaped_chronological_system_update(
        &mut chronological_user_update_content,
        system_update_content,
    );
    translated_chat_messages.push(ChatMessage::User {
        content: chronological_user_update_content,
        images: Vec::new(),
    });
}

fn append_escaped_chronological_system_update(
    chronological_user_update_content: &mut String,
    system_update_content: &str,
) {
    chronological_user_update_content.push_str(CHRONOLOGICAL_SYSTEM_UPDATE_OPENING_TAG);
    for system_update_character in system_update_content.chars() {
        match system_update_character {
            '&' => chronological_user_update_content.push_str("&amp;"),
            '<' => chronological_user_update_content.push_str("&lt;"),
            '>' => chronological_user_update_content.push_str("&gt;"),
            _ => chronological_user_update_content.push(system_update_character),
        }
    }
    chronological_user_update_content.push_str(CHRONOLOGICAL_SYSTEM_UPDATE_CLOSING_TAG);
}

fn translate_images(
    images: Vec<astronomical_rest_contract::OpenAiImageInput>,
) -> Vec<ChatImageInput> {
    images
        .into_iter()
        .map(|image| ChatImageInput {
            mime_type: image.mime_type().to_owned(),
            decoded_bytes: image.decoded_bytes().to_vec(),
        })
        .collect()
}

fn translate_tools(tools: Vec<OpenAiToolDefinitionParts>) -> Vec<ChatToolDefinition> {
    tools
        .into_iter()
        .map(|tool| ChatToolDefinition {
            name: tool.name,
            description: tool.description,
            parameters_json: tool.parameters_json,
        })
        .collect()
}

fn translate_tool_choice(
    tool_choice: OpenAiToolChoiceMode,
) -> Result<ChatToolChoice, OpenAiChatTranslationError> {
    match tool_choice {
        OpenAiToolChoiceMode::Auto => Ok(ChatToolChoice::Auto),
        OpenAiToolChoiceMode::None => Ok(ChatToolChoice::None),
        OpenAiToolChoiceMode::Required => {
            Err(OpenAiChatTranslationError::UnsupportedToolChoice { mode: "required" })
        }
        OpenAiToolChoiceMode::Function { .. } => {
            Err(OpenAiChatTranslationError::UnsupportedToolChoice { mode: "function" })
        }
    }
}

fn translate_thousandths(
    sampling_parameter: Option<f32>,
    parameter_name: &'static str,
) -> Result<Option<u16>, OpenAiChatTranslationError> {
    let Some(sampling_parameter) = sampling_parameter else {
        return Ok(None);
    };
    let scaled_sampling_parameter = sampling_parameter * 1_000.0;
    let rounded_sampling_parameter = scaled_sampling_parameter.round();
    if (scaled_sampling_parameter - rounded_sampling_parameter).abs() > 0.000_1 {
        return Err(OpenAiChatTranslationError::SamplingPrecisionUnsupported {
            parameter_name,
            requested_value: sampling_parameter,
        });
    }
    u16::try_from(rounded_sampling_parameter as u32)
        .map(Some)
        .map_err(
            |_| OpenAiChatTranslationError::SamplingPrecisionUnsupported {
                parameter_name,
                requested_value: sampling_parameter,
            },
        )
}

/// A typed failure while translating public OpenAI data into the worker protocol.
#[derive(Debug, Error)]
pub enum OpenAiChatTranslationError {
    /// Public validation rejected the caller's OpenAI-compatible request.
    #[error("OpenAI chat request validation failed: {0}")]
    PublicValidation(#[source] OpenAiChatCompletionValidationError),
    /// The public generated-token budget did not fit the IPC representation.
    #[error("OpenAI output token count {actual_output_tokens} does not fit the IPC representation")]
    OutputTokenCountTooLarge { actual_output_tokens: u32 },
    /// The thinking-budget token count did not fit the IPC representation.
    #[error("thinking_budget exceeds the maximum representable token count")]
    ThinkingBudgetTooLarge,
    /// The public floating-point sampling value cannot be represented exactly in thousandths.
    #[error("{parameter_name} value {requested_value} cannot be represented in thousandths")]
    SamplingPrecisionUnsupported {
        parameter_name: &'static str,
        requested_value: f32,
    },
    /// A requested tool policy passed through an invalid caller path.
    #[error("tool choice mode '{mode}' is unsupported")]
    UnsupportedToolChoice { mode: &'static str },
    /// The translated data did not meet independent worker IPC bounds.
    #[error("translated chat IPC command failed validation: {0}")]
    IpcValidation(#[source] ChatGenerationValidationError),
}
