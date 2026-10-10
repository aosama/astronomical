//! Context-window admission arithmetic shared by every request path.
//!
//! Input plus output tokens must fit the model's native position range; this
//! check is independent of any loaded tokenizer, so it lives beside the
//! tokenizer without being owned by it.

use super::tokenizer_error::Qwen3_5TokenizerError;

/// Validates combined input and output tokens against the model artifact's
/// native position range. The operator-configured context limit is advisory:
/// crossing it only records a server-side warning so agentic callers that
/// slightly overshoot (for example during compaction) are still served, while
/// the artifact window remains the only hard rejection boundary.
pub fn validate_context_token_count(
    input_token_count: usize,
    maximum_output_tokens: usize,
    hard_maximum_context_tokens: usize,
    advertised_context_tokens: usize,
) -> Result<(), Qwen3_5TokenizerError> {
    let total_context_tokens = input_token_count.checked_add(maximum_output_tokens).ok_or(
        Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens: usize::MAX,
            maximum_total_context_tokens: hard_maximum_context_tokens,
        },
    )?;
    if total_context_tokens > advertised_context_tokens {
        tracing::warn!(
            total_context_tokens,
            advertised_context_tokens,
            "request exceeds the configured context limit; serving anyway because it fits the model artifact context window"
        );
    }
    if total_context_tokens > hard_maximum_context_tokens {
        return Err(Qwen3_5TokenizerError::TotalContextTooLarge {
            actual_total_context_tokens: total_context_tokens,
            maximum_total_context_tokens: hard_maximum_context_tokens,
        });
    }
    Ok(())
}
