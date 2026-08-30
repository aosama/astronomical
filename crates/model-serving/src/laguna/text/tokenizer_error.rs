use thiserror::Error;

/// A typed failure while loading, encoding, or incrementally decoding Laguna tokens.
#[derive(Debug, Error)]
pub enum LagunaTokenizerError {
    #[error("failed to load validated Laguna tokenizer bytes")]
    LoadTokenizer {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to encode the rendered Laguna prompt")]
    EncodePrompt {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to decode generated Laguna tokens")]
    DecodeGeneratedTokens {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error(
        "generated token {generated_token_id} is outside Laguna model vocabulary {model_vocabulary_size}"
    )]
    GeneratedTokenOutOfVocabulary {
        generated_token_id: u32,
        model_vocabulary_size: u32,
    },
    #[error("decoded Laguna text rewrote an already emitted prefix")]
    DecodedTextRewrotePrefix,
}
