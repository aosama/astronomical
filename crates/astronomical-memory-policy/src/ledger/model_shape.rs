//! What a loaded model occupies and could keep.

use super::phase::LedgerPhase;

/// Byte facts read from the model's files once, at load.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelMemoryShape {
    /// Weights that are always in memory: everything that never streams.
    pub core_weight_bytes: u64,
    /// Every weight byte that may either be kept or read from storage when
    /// needed: a sparse model's experts, a dense model's decoder layers.
    pub streamable_weight_bytes: u64,
    /// What the layers being read occupy while a prompt is read.
    pub reading_prompt_stream_slot_bytes: u64,
    /// What the weights being read occupy while a reply is written: far less
    /// for a sparse model, which reads only the experts one token chose.
    pub writing_reply_stream_slot_bytes: u64,
}

impl ModelMemoryShape {
    /// The slots a phase needs when at least one weight has to be read.
    #[must_use]
    pub const fn stream_slot_bytes_for(&self, phase: LedgerPhase) -> u64 {
        match phase {
            LedgerPhase::Idle => 0,
            LedgerPhase::ReadingPrompt { .. } => self.reading_prompt_stream_slot_bytes,
            LedgerPhase::WritingReply => self.writing_reply_stream_slot_bytes,
        }
    }
}

#[cfg(test)]
#[path = "model_shape_tests.rs"]
mod tests;
