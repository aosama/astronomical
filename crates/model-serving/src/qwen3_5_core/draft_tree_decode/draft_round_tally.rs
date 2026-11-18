//! What a reply's draft rounds added up to: how many ran, how many tokens
//! they emitted and how long they took.
//!
//! A round is worth running only when the tokens it keeps outpace the plain
//! steps its time would have bought, and both sides of that move with the
//! model, its memory and the text. The tally is the reply's own evidence of
//! which side won, written once to the log when the reply ends.

use std::time::Duration;

/// Totals over one reply's draft rounds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DraftRoundTally {
    pub(crate) round_count: u64,
    /// Tokens the rounds added to the reply: each round's anchor and the
    /// drafts it kept. A round's own next pick is not counted until the
    /// round after commits it as its anchor.
    pub(crate) emitted_token_count: u64,
    pub(crate) round_nanoseconds: u64,
}

impl DraftRoundTally {
    pub(crate) fn record_round(&mut self, emitted_token_count: usize, round_duration: Duration) {
        self.round_count = self.round_count.saturating_add(1);
        self.emitted_token_count = self
            .emitted_token_count
            .saturating_add(u64::try_from(emitted_token_count).unwrap_or(u64::MAX));
        self.round_nanoseconds = self
            .round_nanoseconds
            .saturating_add(u64::try_from(round_duration.as_nanos()).unwrap_or(u64::MAX));
    }

    /// Tokens a round added on average; `None` before the first round.
    pub(crate) fn tokens_per_round(&self) -> Option<f64> {
        if self.round_count == 0 {
            return None;
        }
        Some(self.emitted_token_count as f64 / self.round_count as f64)
    }

    /// Milliseconds a round took on average; `None` before the first round.
    pub(crate) fn milliseconds_per_round(&self) -> Option<f64> {
        if self.round_count == 0 {
            return None;
        }
        Some(self.round_nanoseconds as f64 / 1e6 / self.round_count as f64)
    }

    /// Tokens a second while drafting; `None` when no time was recorded.
    pub(crate) fn tokens_per_second(&self) -> Option<f64> {
        if self.round_nanoseconds == 0 {
            return None;
        }
        Some(self.emitted_token_count as f64 * 1e9 / self.round_nanoseconds as f64)
    }
}

#[cfg(test)]
#[path = "draft_round_tally_tests.rs"]
mod tests;
