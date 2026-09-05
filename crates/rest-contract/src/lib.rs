#![forbid(unsafe_code)]

mod image_input;
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
    EnforcedStructuredGeneration, OpenAiStructuredOutputs, OpenAiStructuredOutputsValidationError,
    enforced_generation_from_extra_body, guided_grammar_to_enforced_generation,
};
