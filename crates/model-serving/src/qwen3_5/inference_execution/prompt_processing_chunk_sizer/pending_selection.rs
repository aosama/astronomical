//! Retains the exact Qwen selection until its corresponding forward completes.

use crate::{
    PromptProcessingChunkOptimizationContext, PromptProcessingChunkSizeSelectionReason,
    PromptProcessingMeasurementContext,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PendingPromptProcessingChunkSelection {
    pub(super) measurement_context: PromptProcessingMeasurementContext,
    pub(super) selected_candidate_chunk_size_tokens: usize,
    pub(super) chunk_start_token_position: usize,
    pub(super) selection_reason: PromptProcessingChunkSizeSelectionReason,
    pub(super) optimization_context: PromptProcessingChunkOptimizationContext,
}
