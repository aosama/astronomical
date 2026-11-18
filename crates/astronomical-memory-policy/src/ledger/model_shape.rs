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

    /// The same model with a companion model's weights held beside its own
    /// for as long as it is loaded: a drafter that guesses the reply ahead.
    /// Those bytes are never given back, so they count as core weights and
    /// come out of what the model may keep of its streamable weights.
    #[must_use]
    pub const fn with_companion_weight_bytes(mut self, companion_weight_bytes: u64) -> Self {
        self.core_weight_bytes = self
            .core_weight_bytes
            .saturating_add(companion_weight_bytes);
        self
    }

    /// The same model writing `reply_rows_per_step` rows in one reply step
    /// instead of one: a draft round verifies several guessed tokens at once,
    /// and each row may read its own weights from storage in the same step.
    /// Zero rows is read as one, the reply a model always writes.
    #[must_use]
    pub const fn with_reply_rows_per_step(mut self, reply_rows_per_step: u64) -> Self {
        let counted_rows = if reply_rows_per_step == 0 {
            1
        } else {
            reply_rows_per_step
        };
        self.writing_reply_stream_slot_bytes = self
            .writing_reply_stream_slot_bytes
            .saturating_mul(counted_rows);
        self
    }
}

#[cfg(test)]
#[path = "model_shape_tests.rs"]
mod tests;
