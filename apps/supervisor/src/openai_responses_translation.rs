use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatGenerationValidationError, ChatImageInput, ChatMessage,
    ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_rest_contract::{
    OpenAiResponseInputItemParts, OpenAiResponseInputParts, OpenAiResponseToolChoiceParts,
    OpenAiResponsesRequest, OpenAiResponsesRequestParts, OpenAiResponsesValidationError,
};
use thiserror::Error;

const CHRONOLOGICAL_INSTRUCTION_OPENING_TAG: &str = "<system-update>\n";
const CHRONOLOGICAL_INSTRUCTION_CLOSING_TAG: &str = "\n</system-update>";

/// Translates one validated Responses request into the existing worker command.
pub fn translate_openai_responses_request(
    request_id: RequestId,
    request: OpenAiResponsesRequest,
) -> Result<ChatGenerationCommand, OpenAiResponsesTranslationError> {
    let request_parts = request
        .into_parts()
        .map_err(OpenAiResponsesTranslationError::PublicValidation)?;
    translate_openai_responses_request_parts(request_id, request_parts)
}

pub(crate) fn translate_openai_responses_request_parts(
    request_id: RequestId,
    request_parts: OpenAiResponsesRequestParts,
) -> Result<ChatGenerationCommand, OpenAiResponsesTranslationError> {
    let mut chat_messages = Vec::new();
    if let Some(instructions) = request_parts.instructions {
        chat_messages.push(ChatMessage::System {
            content: instructions,
        });
    }
    match request_parts.input {
        OpenAiResponseInputParts::Text(input_text) => chat_messages.push(ChatMessage::User {
            content: input_text,
            images: Vec::new(),
        }),
        OpenAiResponseInputParts::Items(response_input_items) => {
            translate_response_input_items(response_input_items, &mut chat_messages);
        }
    }
    let maximum_output_tokens =
        u16::try_from(request_parts.maximum_output_tokens).map_err(|_| {
            OpenAiResponsesTranslationError::OutputTokenCountTooLarge {
                actual_output_tokens: request_parts.maximum_output_tokens,
            }
        })?;
    let structured_output = request_parts.structured_output;
    let mut chat_generation_command = ChatGenerationCommand {
        request_id,
        model: request_parts.model,
        messages: chat_messages,
        tools: request_parts
            .tools
            .into_iter()
            .map(|function_tool| {
                let parameters_json = serde_json::to_string(&function_tool.parameters)
                    .map_err(OpenAiResponsesTranslationError::ToolSchemaSerialization)?;
                Ok(ChatToolDefinition {
                    name: function_tool.name,
                    description: function_tool.description,
                    parameters_json,
                })
            })
            .collect::<Result<Vec<_>, OpenAiResponsesTranslationError>>()?,
        tool_choice: match request_parts.tool_choice {
            OpenAiResponseToolChoiceParts::Auto => ChatToolChoice::Auto,
            OpenAiResponseToolChoiceParts::None => ChatToolChoice::None,
        },
        settings: ChatGenerationSettings {
            max_output_tokens: maximum_output_tokens,
            temperature_thousandths: translate_thousandths(
                request_parts.temperature,
                "temperature",
            )?,
            top_p_thousandths: translate_thousandths(request_parts.top_p, "top_p")?,
            seed: None,
            // Thinking-enabled Qwen otherwise spends max_output_tokens inside reasoning
            // and never emits the JSON object structured output asked for.
            thinking_budget: request_parts
                .thinking_budget
                .map(u16::try_from)
                .transpose()
                .map_err(|_| OpenAiResponsesTranslationError::ThinkingBudgetTooLarge)?,
        },
        qwen_thinking_channel_seed: None,
    };
    crate::structured_output::apply_structured_output_instruction(
        &mut chat_generation_command.messages,
        structured_output.as_ref(),
    );
    chat_generation_command
        .validate()
        .map_err(OpenAiResponsesTranslationError::IpcValidation)?;
    Ok(chat_generation_command)
}

#[derive(Default)]
struct PendingAssistantMessage {
    content: String,
    reasoning_content: String,
    tool_calls: Vec<ChatAssistantToolCall>,
}

fn translate_response_input_items(
    response_input_items: Vec<OpenAiResponseInputItemParts>,
    chat_messages: &mut Vec<ChatMessage>,
) {
    let mut pending_assistant_message = PendingAssistantMessage::default();
    for response_input_item in response_input_items {
        match response_input_item {
            OpenAiResponseInputItemParts::SystemMessage { content }
            | OpenAiResponseInputItemParts::DeveloperMessage { content } => {
                flush_pending_assistant_message(chat_messages, &mut pending_assistant_message);
                append_instruction(chat_messages, &content);
            }
            OpenAiResponseInputItemParts::UserMessage { content, images } => {
                flush_pending_assistant_message(chat_messages, &mut pending_assistant_message);
                chat_messages.push(ChatMessage::User {
                    content,
                    images: images
                        .into_iter()
                        .map(|image| ChatImageInput {
                            mime_type: image.mime_type().to_owned(),
                            decoded_bytes: image.decoded_bytes().to_vec(),
                        })
                        .collect(),
                });
            }
            OpenAiResponseInputItemParts::AssistantMessage { content } => {
                pending_assistant_message.content.push_str(&content);
            }
            OpenAiResponseInputItemParts::Reasoning { content } => {
                pending_assistant_message
                    .reasoning_content
                    .push_str(&content);
            }
            OpenAiResponseInputItemParts::FunctionCall {
                call_id,
                name,
                arguments_json,
            } => pending_assistant_message
                .tool_calls
                .push(ChatAssistantToolCall {
                    id: call_id,
                    function: ChatAssistantToolFunction {
                        name,
                        arguments_json,
                    },
                }),
            OpenAiResponseInputItemParts::FunctionCallOutput { call_id, output } => {
                flush_pending_assistant_message(chat_messages, &mut pending_assistant_message);
                chat_messages.push(ChatMessage::Tool {
                    tool_call_id: call_id,
                    content: output,
                });
            }
        }
    }
    flush_pending_assistant_message(chat_messages, &mut pending_assistant_message);
}

fn flush_pending_assistant_message(
    chat_messages: &mut Vec<ChatMessage>,
    pending_assistant_message: &mut PendingAssistantMessage,
) {
    if pending_assistant_message.content.is_empty()
        && pending_assistant_message.reasoning_content.is_empty()
        && pending_assistant_message.tool_calls.is_empty()
    {
        return;
    }
    chat_messages.push(ChatMessage::Assistant {
        content: take_non_empty_string(&mut pending_assistant_message.content),
        reasoning_content: take_non_empty_string(&mut pending_assistant_message.reasoning_content),
        tool_calls: std::mem::take(&mut pending_assistant_message.tool_calls),
    });
}

fn take_non_empty_string(string_content: &mut String) -> Option<String> {
    if string_content.is_empty() {
        None
    } else {
        Some(std::mem::take(string_content))
    }
}

fn append_instruction(chat_messages: &mut Vec<ChatMessage>, instruction_content: &str) {
    if chat_messages.is_empty() {
        chat_messages.push(ChatMessage::System {
            content: instruction_content.to_owned(),
        });
        return;
    }
    if let Some(ChatMessage::User {
        content: prior_user_content,
        ..
    }) = chat_messages.last_mut()
    {
        prior_user_content.push('\n');
        append_escaped_instruction(prior_user_content, instruction_content);
        return;
    }
    let mut chronological_instruction = String::new();
    append_escaped_instruction(&mut chronological_instruction, instruction_content);
    chat_messages.push(ChatMessage::User {
        content: chronological_instruction,
        images: Vec::new(),
    });
}

fn append_escaped_instruction(target_content: &mut String, instruction_content: &str) {
    target_content.push_str(CHRONOLOGICAL_INSTRUCTION_OPENING_TAG);
    for instruction_character in instruction_content.chars() {
        match instruction_character {
            '&' => target_content.push_str("&amp;"),
            '<' => target_content.push_str("&lt;"),
            '>' => target_content.push_str("&gt;"),
            _ => target_content.push(instruction_character),
        }
    }
    target_content.push_str(CHRONOLOGICAL_INSTRUCTION_CLOSING_TAG);
}

fn translate_thousandths(
    sampling_parameter: Option<f32>,
    parameter_name: &'static str,
) -> Result<Option<u16>, OpenAiResponsesTranslationError> {
    let Some(sampling_parameter) = sampling_parameter else {
        return Ok(None);
    };
    let scaled_sampling_parameter = sampling_parameter * 1_000.0;
    let rounded_sampling_parameter = scaled_sampling_parameter.round();
    if (scaled_sampling_parameter - rounded_sampling_parameter).abs() > 0.000_1 {
        return Err(
            OpenAiResponsesTranslationError::SamplingPrecisionUnsupported {
                parameter_name,
                requested_value: sampling_parameter,
            },
        );
    }
    u16::try_from(rounded_sampling_parameter as u32)
        .map(Some)
        .map_err(
            |_| OpenAiResponsesTranslationError::SamplingPrecisionUnsupported {
                parameter_name,
                requested_value: sampling_parameter,
            },
        )
}

#[derive(Debug, Error)]
pub enum OpenAiResponsesTranslationError {
    #[error("OpenAI Responses request validation failed: {0}")]
    PublicValidation(#[source] OpenAiResponsesValidationError),
    #[error("OpenAI Responses output token count {actual_output_tokens} does not fit IPC")]
    OutputTokenCountTooLarge { actual_output_tokens: u32 },
    #[error("{parameter_name} value {requested_value} cannot be represented in thousandths")]
    SamplingPrecisionUnsupported {
        parameter_name: &'static str,
        requested_value: f32,
    },
    #[error("Responses function-tool schema could not be serialized: {0}")]
    ToolSchemaSerialization(#[source] serde_json::Error),
    #[error("translated Responses IPC command failed validation: {0}")]
    IpcValidation(#[source] ChatGenerationValidationError),
    #[error("thinking_budget does not fit the worker representation")]
    ThinkingBudgetTooLarge,
}
