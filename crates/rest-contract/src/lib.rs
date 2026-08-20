#![forbid(unsafe_code)]

mod image_input;
mod openai_chat_completion_request;
mod openai_chat_completion_response;
mod openai_chat_types;
mod openai_image_generation_request;
mod openai_image_generation_response;
mod openai_models_and_errors;
mod openai_responses_input;
mod openai_responses_request;
mod openai_responses_response;
mod openai_responses_stream;
mod openai_responses_tools;

pub use image_input::{MAX_OPENAI_IMAGE_BYTES, OpenAiImageInput};
pub use openai_chat_completion_request::{
    DEFAULT_OPENAI_OUTPUT_TOKENS, MAX_OPENAI_OUTPUT_TOKENS, MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH,
    OpenAiChatCompletionRequest, OpenAiChatCompletionRequestParts,
    OpenAiChatCompletionValidationError,
};
pub use openai_chat_completion_response::{
    OpenAiAssistantMessage, OpenAiChatCompletionChoice, OpenAiChatCompletionChunk,
    OpenAiChatCompletionChunkChoice, OpenAiChatCompletionDelta, OpenAiChatCompletionResponse,
    OpenAiFinishReason, OpenAiResponseToolCall, OpenAiTokenUsage, OpenAiToolCallDelta,
    OpenAiToolCallFunctionDelta,
};
pub use openai_chat_types::{
    OpenAiAssistantToolCall, OpenAiAssistantToolCallParts, OpenAiAssistantToolFunction,
    OpenAiChatMessage, OpenAiChatMessageParts, OpenAiContentPart, OpenAiFunctionChoice,
    OpenAiFunctionDefinition, OpenAiMessageContent, OpenAiStopSequences, OpenAiStreamOptions,
    OpenAiToolChoice, OpenAiToolChoiceMode, OpenAiToolDefinition, OpenAiToolDefinitionParts,
    OpenAiToolType,
};
pub use openai_image_generation_request::{
    MAX_OPENAI_IMAGE_DIMENSION_PIXELS, MIN_OPENAI_IMAGE_DIMENSION_PIXELS,
    OpenAiImageGenerationRequest, OpenAiImageGenerationRequestParts,
    OpenAiImageGenerationResponseFormat, OpenAiImageGenerationValidationError,
};
pub use openai_image_generation_response::{
    OpenAiGeneratedImageParts, OpenAiImageGenerationResponse,
};
pub use openai_models_and_errors::{
    OpenAiError, OpenAiErrorResponse, OpenAiImageModelParts, OpenAiModel, OpenAiModelList,
    OpenAiModelParts, OpenAiModelValidationError,
};
pub use openai_responses_input::{
    OpenAiResponseInput, OpenAiResponseInputItem, OpenAiResponseInputItemParts,
    OpenAiResponseInputParts,
};
pub use openai_responses_request::{
    OpenAiResponsesRequest, OpenAiResponsesRequestParts, OpenAiResponsesValidationError,
};
pub use openai_responses_response::{
    OpenAiResponse, OpenAiResponseError, OpenAiResponseFunctionTool,
    OpenAiResponseIncompleteDetails, OpenAiResponseItemStatus, OpenAiResponseOutputContent,
    OpenAiResponseOutputItem, OpenAiResponseReasoningContent, OpenAiResponseReasoningSummary,
    OpenAiResponseRequestConfiguration, OpenAiResponseStatus, OpenAiResponseUsage,
};
pub use openai_responses_stream::OpenAiResponseStreamEvent;
pub use openai_responses_tools::{
    OpenAiResponseToolChoice, OpenAiResponseToolChoiceParts, OpenAiResponseToolDefinition,
    OpenAiResponseToolDefinitionParts,
};
