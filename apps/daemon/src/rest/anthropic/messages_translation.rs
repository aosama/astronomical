//! Pure translation of a validated Anthropic Messages request onto the
//! worker's chat-generation command, the same command the OpenAI surfaces
//! build, so both protocols share one generation path.

use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatAssistantToolFunction, ChatGenerationCommand,
    ChatGenerationSettings, ChatGenerationValidationError, ChatImageInput, ChatMessage,
    ChatToolChoice, ChatToolDefinition, RequestId,
};
use astronomical_rest_contract::{
    AnthropicMessageParts, AnthropicMessagesRequestParts, AnthropicToolChoiceParts,
    AnthropicToolDefinitionParts, AnthropicToolUseParts, AnthropicUserSegmentParts,
    OpenAiImageInput,
};
use thiserror::Error;

use crate::rest::chronological_instruction::append_instruction;

/// Anthropic flags a failed tool run with `is_error`; the chat template has
/// no such field, so the flag reaches the model as this prefix instead.
const TOOL_ERROR_PREFIX: &str = "Error: ";

pub(crate) fn translate_anthropic_messages_request_parts(
    request_id: RequestId,
    request_parts: AnthropicMessagesRequestParts,
) -> Result<ChatGenerationCommand, AnthropicMessagesTranslationError> {
    let mut chat_messages = Vec::new();
    if let Some(system_text) = request_parts.system {
        chat_messages.push(ChatMessage::System {
            content: system_text,
        });
    }
    for message_parts in request_parts.messages {
        append_chat_messages(message_parts, &mut chat_messages);
    }
    let mut tools = Vec::with_capacity(request_parts.tools.len());
    for tool_definition in request_parts.tools {
        tools.push(chat_tool_definition(tool_definition)?);
    }
    let chat_generation_command = ChatGenerationCommand {
        request_id,
        model: request_parts.model,
        messages: chat_messages,
        tools,
        tool_choice: match request_parts.tool_choice {
            AnthropicToolChoiceParts::Auto => ChatToolChoice::Auto,
            AnthropicToolChoiceParts::None => ChatToolChoice::None,
        },
        settings: generation_settings(
            request_parts.maximum_output_tokens,
            request_parts.temperature,
            request_parts.top_p,
            request_parts.thinking_budget,
        )?,
        structured_generation: None,
    };
    chat_generation_command
        .validate()
        .map_err(AnthropicMessagesTranslationError::IpcValidation)?;
    Ok(chat_generation_command)
}

fn generation_settings(
    maximum_output_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    thinking_budget: Option<u32>,
) -> Result<ChatGenerationSettings, AnthropicMessagesTranslationError> {
    let Ok(max_output_tokens) = u16::try_from(maximum_output_tokens) else {
        return Err(
            AnthropicMessagesTranslationError::OutputTokenCountTooLarge {
                actual_output_tokens: maximum_output_tokens,
            },
        );
    };
    let thinking_budget = match thinking_budget {
        None => None,
        Some(thinking_budget) => match u16::try_from(thinking_budget) {
            Ok(thinking_budget) => Some(thinking_budget),
            Err(_) => return Err(AnthropicMessagesTranslationError::ThinkingBudgetTooLarge),
        },
    };
    Ok(ChatGenerationSettings {
        max_output_tokens,
        temperature_thousandths: temperature.map(unit_interval_thousandths),
        top_p_thousandths: top_p.map(unit_interval_thousandths),
        // The Messages protocol has no repetition penalty; the model's
        // configured default always applies.
        repetition_penalty_thousandths: None,
        seed: None,
        thinking_budget,
    })
}

/// Rounds a contract-validated 0..=1 sampling value to thousandths. Anthropic
/// clients send values such as 0.7 that are not exact in binary, so this
/// rounds instead of rejecting the imprecision the OpenAI surfaces reject.
fn unit_interval_thousandths(unit_interval_value: f32) -> u16 {
    (unit_interval_value * 1_000.0).round() as u16
}

fn chat_tool_definition(
    tool_definition: AnthropicToolDefinitionParts,
) -> Result<ChatToolDefinition, AnthropicMessagesTranslationError> {
    let parameters_json = serde_json::to_string(&tool_definition.input_schema)
        .map_err(AnthropicMessagesTranslationError::ToolSchemaSerialization)?;
    Ok(ChatToolDefinition {
        name: tool_definition.name,
        description: tool_definition.description,
        parameters_json,
    })
}

fn append_chat_messages(
    message_parts: AnthropicMessageParts,
    chat_messages: &mut Vec<ChatMessage>,
) {
    match message_parts {
        AnthropicMessageParts::User { segments } => append_user_turn(segments, chat_messages),
        AnthropicMessageParts::Assistant {
            text,
            thinking,
            tool_uses,
        } => {
            let mut tool_calls = Vec::with_capacity(tool_uses.len());
            for tool_use in tool_uses {
                tool_calls.push(chat_assistant_tool_call(tool_use));
            }
            chat_messages.push(ChatMessage::Assistant {
                content: non_empty_text(text),
                reasoning_content: non_empty_text(thinking),
                tool_calls,
            });
        }
        // Tool references were checked against the declared tools by the
        // contract; every declared tool is already visible to the model.
        AnthropicMessageParts::System { text, .. } => {
            if !text.is_empty() {
                append_instruction(chat_messages, &text);
            }
        }
    }
}

/// The text and images a user turn has accumulated since its last tool result.
#[derive(Default)]
struct PendingUserMessage {
    texts: Vec<String>,
    images: Vec<ChatImageInput>,
}

/// One Anthropic user turn becomes chat messages in block order: each
/// `tool_result` is its own tool message, and the text and images around
/// them gather into user messages. A tool result's images have no place on a
/// tool message, so they follow it in the next user message, where a vision
/// model still sees them.
fn append_user_turn(
    segments: Vec<AnthropicUserSegmentParts>,
    chat_messages: &mut Vec<ChatMessage>,
) {
    let mut pending_user_message = PendingUserMessage::default();
    for segment in segments {
        match segment {
            AnthropicUserSegmentParts::Text(text) => pending_user_message.texts.push(text),
            AnthropicUserSegmentParts::Image(image) => {
                pending_user_message.images.push(chat_image_input(&image));
            }
            AnthropicUserSegmentParts::ToolResult {
                tool_use_id,
                text,
                images,
                is_error,
            } => {
                flush_pending_user_message(&mut pending_user_message, chat_messages);
                chat_messages.push(ChatMessage::Tool {
                    tool_call_id: tool_use_id,
                    content: tool_result_content(text, is_error),
                });
                for image in images {
                    pending_user_message.images.push(chat_image_input(&image));
                }
            }
        }
    }
    flush_pending_user_message(&mut pending_user_message, chat_messages);
}

fn flush_pending_user_message(
    pending_user_message: &mut PendingUserMessage,
    chat_messages: &mut Vec<ChatMessage>,
) {
    if pending_user_message.texts.is_empty() && pending_user_message.images.is_empty() {
        return;
    }
    chat_messages.push(ChatMessage::User {
        content: std::mem::take(&mut pending_user_message.texts).join("\n"),
        images: std::mem::take(&mut pending_user_message.images),
    });
}

fn tool_result_content(text: String, is_error: bool) -> String {
    if is_error {
        format!("{TOOL_ERROR_PREFIX}{text}")
    } else {
        text
    }
}

fn chat_image_input(image: &OpenAiImageInput) -> ChatImageInput {
    ChatImageInput {
        mime_type: image.mime_type().to_owned(),
        decoded_bytes: image.decoded_bytes().to_vec(),
    }
}

fn chat_assistant_tool_call(tool_use: AnthropicToolUseParts) -> ChatAssistantToolCall {
    ChatAssistantToolCall {
        id: tool_use.id,
        function: ChatAssistantToolFunction {
            name: tool_use.name,
            arguments_json: tool_use.input_json,
        },
    }
}

fn non_empty_text(text: String) -> Option<String> {
    if text.is_empty() { None } else { Some(text) }
}

#[derive(Debug, Error)]
pub(crate) enum AnthropicMessagesTranslationError {
    #[error("max_tokens {actual_output_tokens} does not fit the worker representation")]
    OutputTokenCountTooLarge { actual_output_tokens: u32 },
    #[error("thinking.budget_tokens does not fit the worker representation")]
    ThinkingBudgetTooLarge,
    #[error("tool input_schema could not be serialized: {0}")]
    ToolSchemaSerialization(#[source] serde_json::Error),
    #[error("{0}")]
    IpcValidation(#[source] ChatGenerationValidationError),
}
