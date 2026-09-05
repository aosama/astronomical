//! Bounded ModernBERT tokenization over the pinned HF tokenizer.
//!
//! Tokenization must happen before any lazy graph construction so an
//! out-of-context request fails before GPU work is submitted.

use astronomical_ipc_protocol::EmbeddingsFailureReason;
use tokenizers::Tokenizer;

use crate::modernbert::configuration::ModernBertConfiguration;

/// Token identifiers for one encoded text input.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedEmbeddingInput {
    pub token_ids: Vec<u32>,
}

/// Encodes one text input or fails with a bounded request failure.
pub fn encode_embedding_input(
    tokenizer: &Tokenizer,
    configuration: &ModernBertConfiguration,
    input_text: &str,
) -> Result<EncodedEmbeddingInput, EmbeddingsFailureReason> {
    // GPU evidence: enabling the tokenizer post-processor (CLS/SEP) collapsed
    // short-text discrimination on this 8-bit artifact (Romeo lines vs finance).
    // First slice encodes content tokens only; a later oracle slice can match
    // HuggingFace special-token behavior with a reference vector.
    let encoding = tokenizer
        .encode(input_text, false)
        .map_err(|tokenizer_error| EmbeddingsFailureReason::InvalidRequest {
            reason: format!("embedding input failed tokenization: {tokenizer_error}")
                .chars()
                .take(256)
                .collect(),
        })?;
    let token_ids = encoding.get_ids().to_vec();
    if token_ids.len() as u64 > u64::from(configuration.maximum_position_count) {
        return Err(EmbeddingsFailureReason::ContextLengthExceeded {
            actual_total_context_tokens: token_ids.len() as u32,
            maximum_context_tokens: configuration.maximum_position_count,
        });
    }
    if token_ids.is_empty() {
        return Err(EmbeddingsFailureReason::InvalidRequest {
            reason: "embedding input encoded to zero tokens".to_owned(),
        });
    }
    Ok(EncodedEmbeddingInput { token_ids })
}
