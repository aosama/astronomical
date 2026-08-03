use astronomical_rest_contract::{
    MAX_OPENAI_OUTPUT_TOKENS, OpenAiChatCompletionRequest, OpenAiChatCompletionValidationError,
    OpenAiChatMessageParts, OpenAiToolChoiceMode,
};
use serde_json::json;

mod image_content;
mod option_validation;
mod standard_request;
mod transport_boundaries;
