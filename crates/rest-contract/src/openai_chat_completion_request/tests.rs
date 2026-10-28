//! Wire-contract tests for the chat completion request DTO.
//!
//! Moved from the integration lane (issue #1610): every scenario exercises
//! exactly this module's parse-and-validate surface with literal JSON and
//! crosses no process, socket, filesystem, or model boundary.

use crate::{
    MAX_OPENAI_OUTPUT_TOKENS, OpenAiChatCompletionRequest, OpenAiChatCompletionValidationError,
    OpenAiChatMessageParts, OpenAiToolChoiceMode,
};
use serde_json::json;

mod history_compatibility;
mod image_content;
mod option_validation;
mod standard_request;
mod thinking_budget_spellings;
mod tool_strict_field;
mod transport_boundaries;
