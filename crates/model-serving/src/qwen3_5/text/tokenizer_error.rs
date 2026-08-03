use astronomical_ipc_protocol::ChatGenerationValidationError;
use thiserror::Error;

use super::{
    Qwen3_5ImageProcessingError, Qwen3_5PromptError, token_ids::Qwen3_5TokenDiscoveryError,
};

/// A bounded failure while loading or using the pinned tokenizer.
#[derive(Debug, Error)]
pub enum Qwen3_5TokenizerError {
    #[error("validated artifact does not retain tokenizer.json bytes")]
    MissingValidatedTokenizer,
    #[error(
        "model output-token budget {max_output_tokens} exceeds context window {context_window}"
    )]
    ModelOutputBudgetExceedsContextWindow {
        context_window: u32,
        max_output_tokens: u32,
    },
    #[error("failed to load captured tokenizer bytes")]
    LoadTokenizer {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("tokenizer vocabulary is too large: {actual_vocabulary_size}")]
    TokenizerVocabularyTooLarge { actual_vocabulary_size: usize },
    #[error(
        "special token '{token_content}' has ID {actual_token_id:?}, expected {expected_token_id}"
    )]
    SpecialTokenMismatch {
        token_content: &'static str,
        expected_token_id: u32,
        actual_token_id: Option<u32>,
    },
    #[error("failed to encode the rendered prompt")]
    EncodePrompt {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error(
        "generated token {generated_token_id} is unavailable in model vocabulary {model_vocabulary_size}"
    )]
    GeneratedTokenOutOfVocabulary {
        generated_token_id: u32,
        model_vocabulary_size: u32,
    },
    #[error("failed to decode generated tokens")]
    DecodeGeneratedTokens {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("decoded text rewrote an already emitted prefix")]
    DecodedTextRewrotePrefix,
    #[error("invalid structured chat command")]
    InvalidChatCommand(#[source] ChatGenerationValidationError),
    #[error("structured chat model '{actual_model_id}' does not match the loaded model")]
    ModelIdMismatch { actual_model_id: String },
    #[error("failed to render the fixed chat prompt")]
    RenderPrompt(#[source] Qwen3_5PromptError),
    #[error("failed to process chat image input through the vision pipeline: {0}")]
    ImageProcessing(#[source] Qwen3_5ImageProcessingError),
    #[error("the selected model supports text input only and cannot process images")]
    ImageInputUnsupported,
    #[error(
        "context has {actual_total_context_tokens} tokens, exceeding {maximum_total_context_tokens}"
    )]
    TotalContextTooLarge {
        actual_total_context_tokens: usize,
        maximum_total_context_tokens: usize,
    },
    #[error("failed to discover special token IDs from tokenizer")]
    DiscoverTokenIds {
        #[source]
        source: Qwen3_5TokenDiscoveryError,
    },
}
