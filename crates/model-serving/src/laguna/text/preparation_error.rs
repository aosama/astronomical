use astronomical_ipc_protocol::ChatGenerationValidationError;
use thiserror::Error;

use super::{LagunaOutputParserError, LagunaPromptRendererError, LagunaTokenizerError};

/// A request-local failure before Laguna inference begins.
#[derive(Debug, Error)]
pub enum LagunaPreparationError {
    #[error(
        "structured chat model '{actual_model_id}' does not match loaded Laguna model '{expected_model_id}'"
    )]
    ModelIdMismatch {
        expected_model_id: String,
        actual_model_id: String,
    },
    #[error("the selected Laguna model supports text input only and cannot process images")]
    ImageInputUnsupported,
    #[error("invalid structured Laguna chat command")]
    InvalidChatCommand(#[source] ChatGenerationValidationError),
    #[error("failed to render the fixed Laguna prompt")]
    PromptRendering(#[source] LagunaPromptRendererError),
    #[error("failed to tokenize the Laguna prompt")]
    PromptTokenization(#[source] LagunaTokenizerError),
    #[error("context has {actual_context_tokens} tokens, exceeding {maximum_context_tokens}")]
    ContextLengthExceeded {
        actual_context_tokens: usize,
        maximum_context_tokens: u32,
    },
    #[error("failed to initialize Laguna output parsing")]
    OutputParserInitialization(#[source] LagunaOutputParserError),
}
