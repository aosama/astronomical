//! What drafting did for one reply, and summed over a session.

use serde::{Deserialize, Serialize};

/// What drafting did for one reply: how many rounds it ran, how many words
/// the drafter guessed, and how many of those the model kept as its own.
/// A reply that never drafted reports none of this.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestDraftingStatistics {
    pub round_count: u32,
    pub drafted_token_count: u32,
    pub accepted_token_count: u32,
}

impl RequestDraftingStatistics {
    /// No round run yet: where a session's tally starts.
    pub const NONE: Self = Self {
        round_count: 0,
        drafted_token_count: 0,
        accepted_token_count: 0,
    };

    /// Counts one round's guesses and the ones kept.
    pub fn record_round(&mut self, drafted_token_count: u32, accepted_token_count: u32) {
        self.round_count = self.round_count.saturating_add(1);
        self.drafted_token_count = self.drafted_token_count.saturating_add(drafted_token_count);
        self.accepted_token_count = self
            .accepted_token_count
            .saturating_add(accepted_token_count);
    }

    /// Moves kept guesses that never reached the reply, because it ended
    /// partway through a kept chain, to the thrown-away side, so the kept
    /// count never exceeds the words the reply holds.
    pub fn withdraw_unsent_accepted_tokens(&mut self, unsent_accepted_token_count: u32) {
        self.accepted_token_count = self
            .accepted_token_count
            .saturating_sub(unsent_accepted_token_count);
    }

    /// Guesses the model did not keep.
    #[must_use]
    pub const fn rejected_token_count(&self) -> u32 {
        self.drafted_token_count
            .saturating_sub(self.accepted_token_count)
    }

    /// Adds another reply's figures, for session totals.
    pub fn accumulate(&mut self, other_reply: &Self) {
        self.round_count = self.round_count.saturating_add(other_reply.round_count);
        self.drafted_token_count = self
            .drafted_token_count
            .saturating_add(other_reply.drafted_token_count);
        self.accepted_token_count = self
            .accepted_token_count
            .saturating_add(other_reply.accepted_token_count);
    }

    /// Whether any round ran; a reply that never drafted reports nothing.
    #[must_use]
    pub const fn has_drafted(&self) -> bool {
        self.round_count > 0
    }
}

#[cfg(test)]
#[path = "request_drafting_statistics_tests.rs"]
mod tests;
