/// Prompt-processing mode selected for one attempted Qwen3.5 chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5SpeculativePrefillChunckMode {
    /// Execute the ordinary target-only path for a request without an optional session.
    OrdinaryTarget,
    /// Execute only the target model before terminal additional-state capture.
    TargetOnlyPrefix,
    /// Retain target hidden rows and initialize private additional history once.
    TerminalAdditionalHistoryCapture,
}

/// Selects whether one prompt chunk may initialize private additional history.
///
/// The final prompt token remains reserved for generation startup, so a
/// terminal prefill chunk ends at `final_prompt_index` rather than at the
/// input token vector's length.
#[must_use]
pub const fn qwen3_5_speculative_prefill_chunck_mode(
    has_optional_prediction_session: bool,
    prefill_end: usize,
    final_prompt_index: usize,
) -> Qwen3_5SpeculativePrefillChunckMode {
    if !has_optional_prediction_session {
        return Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget;
    }
    if prefill_end == final_prompt_index {
        Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
    } else {
        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix
    }
}
