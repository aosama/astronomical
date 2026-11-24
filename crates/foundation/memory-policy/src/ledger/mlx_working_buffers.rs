//! The measured MLX working buffers of one layer, by phase.
//!
//! MLX working buffers are the scratch space one layer needs while it processes
//! one span. It is measured directly: the peak MLX reports for one layer
//! step, less what was held when the step began. It is never derived by
//! subtracting estimates from each other, which is where issue #2218 came
//! from.

use std::collections::BTreeMap;

use super::phase::LedgerPhase;

/// What to set aside for one layer's MLX working buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlxWorkingBufferReserve {
    /// A measurement covers this phase.
    Measured(u64),
    /// No measurement covers this phase yet. The ledger then keeps no
    /// weights, because an unknown need cannot be planned around.
    Unmeasured,
}

/// The largest MLX working buffers seen for each kind of layer step.
///
/// Values only grow. One light request must not erase what a heavier one
/// proved was needed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MlxWorkingBufferMeasurements {
    reading_prompt_bytes_by_span_token_count: BTreeMap<u32, u64>,
    writing_reply_bytes: Option<u64>,
}

impl MlxWorkingBufferMeasurements {
    /// What to set aside for one layer step of `phase`.
    ///
    /// A span is covered by the measurement of the smallest span at least as
    /// large, because a larger span never needs less. A span larger than any
    /// measured one is unmeasured: scaling a small measurement up multiplied
    /// its fixed overhead into reserves several times the limit (issues #623
    /// and #690), so nothing is extrapolated upward.
    #[must_use]
    pub fn reserve_for(&self, phase: LedgerPhase) -> MlxWorkingBufferReserve {
        match phase {
            LedgerPhase::Idle => MlxWorkingBufferReserve::Measured(0),
            LedgerPhase::WritingReply => match self.writing_reply_bytes {
                Some(writing_reply_bytes) => MlxWorkingBufferReserve::Measured(writing_reply_bytes),
                None => MlxWorkingBufferReserve::Unmeasured,
            },
            LedgerPhase::ReadingPrompt { span_token_count } => {
                self.reading_prompt_reserve(span_token_count)
            }
        }
    }

    fn reading_prompt_reserve(&self, span_token_count: u32) -> MlxWorkingBufferReserve {
        let Some((_, covering_span_bytes)) = self
            .reading_prompt_bytes_by_span_token_count
            .range(span_token_count..)
            .next()
        else {
            return MlxWorkingBufferReserve::Unmeasured;
        };
        // A smaller span that measured more than the covering one is noise in
        // the larger measurement, not a reason to reserve less.
        let mut reserve_bytes = *covering_span_bytes;
        for (_, smaller_span_bytes) in self
            .reading_prompt_bytes_by_span_token_count
            .range(..span_token_count)
        {
            reserve_bytes = reserve_bytes.max(*smaller_span_bytes);
        }
        MlxWorkingBufferReserve::Measured(reserve_bytes)
    }

    /// Records one measured layer step. Returns whether it raised what is
    /// known for that step, which is the only case that changes any account.
    pub fn record(&mut self, phase: LedgerPhase, observed_mlx_working_buffer_bytes: u64) -> bool {
        let known_bytes = match phase {
            LedgerPhase::Idle => return false,
            LedgerPhase::WritingReply => self.writing_reply_bytes.get_or_insert(0),
            LedgerPhase::ReadingPrompt { span_token_count } => self
                .reading_prompt_bytes_by_span_token_count
                .entry(span_token_count)
                .or_insert(0),
        };
        // Reaching this point has already made the step measured, even when
        // the measurement is zero and raises nothing.
        if observed_mlx_working_buffer_bytes > *known_bytes {
            *known_bytes = observed_mlx_working_buffer_bytes;
            return true;
        }
        false
    }

    /// Whether any layer step of `phase` has been measured.
    #[must_use]
    pub fn covers(&self, phase: LedgerPhase) -> bool {
        self.reserve_for(phase) != MlxWorkingBufferReserve::Unmeasured
    }
}

#[cfg(test)]
#[path = "mlx_working_buffers_tests.rs"]
mod tests;
