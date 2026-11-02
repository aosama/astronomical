//! Typed OpenAI-compatible and Anthropic-compatible request/response
//! contracts for Astronomical's loopback REST surface.
//!
//! # Test tiers
//!
//! Every test in this crate is a unit test and lives beside the types it
//! exercises, in `#[cfg(test)]` submodules of `src/` (issue #1610). The
//! crate has no integration lane; the `rest_api_tests` target CI selects
//! resolves from the daemon crate, which drives the full application stack.

#![forbid(unsafe_code)]

mod anthropic_count_tokens;
mod anthropic_messages_content;
mod anthropic_messages_request;
mod anthropic_messages_response;
mod anthropic_model_listing;
mod image_input;
mod json_nesting_depth;
mod openai_chat_completion_request;
mod openai_chat_completion_response;
mod openai_chat_types;
mod openai_embeddings_request;
mod openai_embeddings_response;
mod openai_image_generation_request;
mod openai_image_generation_response;
mod openai_models_and_errors;
mod openai_response_format;
mod openai_responses_input;
mod openai_responses_request;
mod openai_responses_response;
mod openai_responses_stream;
mod openai_responses_tools;
mod openai_structured_outputs;
mod request_budget;
mod thinking_controls;
mod tool_name_rule;

pub use anthropic_count_tokens::{AnthropicCountTokensRequest, AnthropicTokenCount};
pub use anthropic_messages_content::{
    AnthropicImageSource, AnthropicInputContentBlock, AnthropicInputMessage,
    AnthropicMessageContent, AnthropicMessageParts, AnthropicSystemBlock, AnthropicSystemPrompt,
    AnthropicToolAddition, AnthropicToolUseParts, AnthropicUserSegmentParts,
};
pub use anthropic_messages_request::{
    AnthropicMessagesRequest, AnthropicMessagesRequestParts, AnthropicMessagesValidationError,
    AnthropicOutputConfiguration, AnthropicThinkingConfiguration, AnthropicToolChoice,
    AnthropicToolChoiceParts, AnthropicToolDefinition, AnthropicToolDefinitionParts,
};
pub use anthropic_messages_response::{
    AnthropicContentBlockDelta, AnthropicError, AnthropicErrorResponse, AnthropicErrorType,
    AnthropicMessage, AnthropicMessageDeltaBody, AnthropicMessageStreamEvent,
    AnthropicOutputContentBlock, AnthropicStopReason, AnthropicUsage,
};
pub use anthropic_model_listing::{AnthropicModelIdentity, AnthropicModelListPage};
pub use image_input::DecodedImageInput;
pub use openai_chat_completion_request::{
    DEFAULT_OPENAI_OUTPUT_TOKENS, OpenAiChatCompletionRequest, OpenAiChatCompletionRequestParts,
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
pub use openai_embeddings_request::{
    OpenAiEmbeddingEncodingFormat, OpenAiEmbeddingInput, OpenAiEmbeddingsRequest,
    OpenAiEmbeddingsRequestParts, OpenAiEmbeddingsValidationError,
};
pub use openai_embeddings_response::{
    OpenAiEmbedding, OpenAiEmbeddingVector, OpenAiEmbeddingsResponse,
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
    OpenAiEmbeddingModelParts, OpenAiError, OpenAiErrorResponse, OpenAiImageModelParts,
    OpenAiModel, OpenAiModelList, OpenAiModelParts, OpenAiModelValidationError,
};
pub use openai_response_format::{
    MAX_STRUCTURED_OUTPUT_SCHEMA_BYTES, OpenAiJsonSchemaSpec, OpenAiResponseFormat,
    OpenAiStructuredOutput, OpenAiStructuredOutputValidationError,
    STRUCTURED_OUTPUT_ENFORCEMENT_LOGITS_MASK, UNENFORCED_RESPONSE_FORMAT_WARNING,
    UNENFORCED_STRICT_RESPONSE_FORMAT_WARNING, compact_extracted_json_text,
    extract_json_value_from_text, merge_structured_output_requests,
    structured_output_from_responses_text_format,
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
pub use openai_structured_outputs::{
    EnforcedStructuredGeneration, MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES, OpenAiStructuredOutputs,
    OpenAiStructuredOutputsValidationError, enforced_generation_from_extra_body,
    enforced_generation_from_json_schema, guided_grammar_to_enforced_generation,
};
pub use request_budget::{
    MAX_IMAGE_DECODED_BYTES, MAX_IMAGE_PARTS_PER_REQUEST, MAX_INLINE_MEDIA_BYTES,
    MAX_OUTPUT_TOKENS, MAX_REQUEST_BODY_BYTES, MAX_TOOL_SCHEMA_NESTING_DEPTH,
    base64_wire_byte_count,
};
pub use thinking_controls::{
    ChatTemplateKwargsRequestObject, ReasoningRequestObject, ThinkingControls,
    ThinkingControlsError, ThinkingControlsInputs,
};
