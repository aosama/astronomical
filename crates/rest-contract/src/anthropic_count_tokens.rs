//! The Anthropic token-counting request (`POST /v1/messages/count_tokens`)
//! and its response. A count request carries a Messages conversation without
//! the generation controls, and it validates through the same helpers as
//! `POST /v1/messages`, so a count never accepts a conversation the Messages
//! endpoint would reject, nor rejects one it would accept.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::anthropic_messages_request::{
    message_parts, resolve_thinking_budget, resolve_thinking_display, resolve_tool_choice,
    tool_definition_parts, validate_tool_references,
};
use crate::{
    AnthropicInputMessage, AnthropicMessagesRequestParts, AnthropicMessagesValidationError,
    AnthropicOutputConfiguration, AnthropicSystemPrompt, AnthropicThinkingConfiguration,
    AnthropicToolChoice, AnthropicToolDefinition,
};

/// The smallest output budget the worker command accepts. A count never
/// generates, so this exists only to satisfy command validation.
const COUNT_ONLY_OUTPUT_TOKENS: u32 = 1;

/// One token-counting request. Unknown fields are rejected, as on the
/// Messages endpoint; `context_management` is absorbed for the same reason.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicCountTokensRequest {
    model: String,
    messages: Vec<AnthropicInputMessage>,
    #[serde(default)]
    system: Option<AnthropicSystemPrompt>,
    #[serde(default)]
    tools: Vec<AnthropicToolDefinition>,
    #[serde(default)]
    tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    thinking: Option<AnthropicThinkingConfiguration>,
    #[serde(default)]
    output_config: Option<AnthropicOutputConfiguration>,
    #[serde(default)]
    context_management: Option<Value>,
    #[serde(flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

impl AnthropicCountTokensRequest {
    /// Validates and consumes this request into the same parts a Messages
    /// request produces. Thinking still validates and shapes the prompt
    /// template, because the template renders differently with thinking off.
    pub fn into_parts(
        self,
    ) -> Result<AnthropicMessagesRequestParts, AnthropicMessagesValidationError> {
        if let Some((field_name, _)) = self.unknown_fields.first_key_value() {
            return Err(AnthropicMessagesValidationError::UnknownField {
                field_name: field_name.clone(),
            });
        }
        if self.model.is_empty() {
            return Err(AnthropicMessagesValidationError::EmptyModel);
        }
        if self.messages.is_empty() {
            return Err(AnthropicMessagesValidationError::EmptyMessages);
        }
        let thinking_budget =
            resolve_thinking_budget(self.thinking.as_ref(), self.output_config.as_ref())?;
        let is_thinking_display_omitted = resolve_thinking_display(self.thinking.as_ref())?;
        let tool_choice = resolve_tool_choice(self.tool_choice.as_ref())?;
        let tools = tool_definition_parts(self.tools)?;
        let system = self
            .system
            .map(AnthropicSystemPrompt::into_text)
            .transpose()?;
        let messages = message_parts(self.messages)?;
        validate_tool_references(&messages, &tools)?;
        Ok(AnthropicMessagesRequestParts {
            model: self.model,
            system,
            messages,
            tools,
            tool_choice,
            maximum_output_tokens: COUNT_ONLY_OUTPUT_TOKENS,
            temperature: None,
            top_p: None,
            stream: false,
            thinking_budget,
            is_thinking_display_omitted,
        })
    }
}

/// The token-counting response: the prompt tokens the conversation would
/// prefill, prompt-cache hits included, because a count runs no cache lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct AnthropicTokenCount {
    pub input_tokens: u32,
}

#[cfg(test)]
mod tests;
