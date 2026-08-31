//! Classifies ordinary, sparse, and optional-history prompt chunks.
//!
//! This module contains no model or MLX state. Target-only prefill consumes every
//! prompt token and samples the first output from the last chunk. Optional
//! prediction keeps the last prompt token for generation kickoff so predictor
//! history can shift onto it; only the chunk that reaches that reserved token
//! may initialize private history.

/// Prompt-processing mode selected for one attempted Qwen3.5 chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5SpeculativePrefillChunkMode {
    /// Execute the ordinary target-only path.
    ///
    /// No optional prediction session exists, so there is no private history to
    /// initialize and every chunk follows the standard target path.
    OrdinaryTarget,
    /// Execute only the target model for a nonterminal prefix chunk.
    ///
    /// An optional prediction session exists, but this chunk does not reach the
    /// terminal prefill boundary. Capturing history now would initialize it from
    /// an incomplete prompt.
    TargetOnlyPrefix,
    /// Retain target hidden rows and initialize private additional history once.
    ///
    /// This is the sole chunk ending immediately before generation startup. Its
    /// target hidden rows provide the complete prompt history required by the
    /// optional predictor.
    TerminalAdditionalHistoryCapture,
}

/// Selects whether one prompt chunk may initialize private additional history.
///
/// Exclusive end of target prefill for this request.
///
/// Optional prediction reserves the last prompt token for generation kickoff.
/// Target-only prefill consumes every prompt token.
#[must_use]
pub const fn qwen3_5_prompt_prefill_end_exclusive(
    prompt_token_count: usize,
    has_optional_prediction_session: bool,
) -> usize {
    if has_optional_prediction_session {
        prompt_token_count.saturating_sub(1)
    } else {
        prompt_token_count
    }
}

/// The optional-prediction terminal chunk ends at `final_prompt_index` because
/// that request reserves the last prompt token for generation kickoff.
#[must_use]
pub const fn qwen3_5_speculative_prefill_chunk_mode(
    has_optional_prediction_session: bool,
    prefill_end: usize,
    final_prompt_index: usize,
) -> Qwen3_5SpeculativePrefillChunkMode {
    // The absence of a session is the strongest condition: terminal position is
    // irrelevant when no additional state owner exists.
    if !has_optional_prediction_session {
        return Qwen3_5SpeculativePrefillChunkMode::OrdinaryTarget;
    }
    // `final_prompt_index` is deliberately an index, not the token-vector length.
    // Equality means this chunk has consumed every token except the one reserved
    // for the generation-kickoff forward.
    if prefill_end == final_prompt_index {
        Qwen3_5SpeculativePrefillChunkMode::TerminalAdditionalHistoryCapture
    } else {
        // Earlier chunks must not initialize history from a partial prompt.
        Qwen3_5SpeculativePrefillChunkMode::TargetOnlyPrefix
    }
}
