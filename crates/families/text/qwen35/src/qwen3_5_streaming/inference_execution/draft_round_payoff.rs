//! Whether a reply's draft rounds are paying, decided inside the reply.
//!
//! A hosted drafter must never make a reply slower than it would be
//! without one (issue #2330), and whether rounds pay depends on what is
//! being written: on a 35B-A3B at a 16 GB limit they write real source 29%
//! faster (64.3 against 49.7 tokens a second) and an essay 3 to 9% slower.
//! A round costs the same whatever it keeps, a draft and a several-row
//! pass, so what separates the two is the tokens a round decides: 2.45 on
//! the source, 1.93 on the essay, with the break-even near 2.1.
//!
//! So a reply counts the tokens its rounds decide and compares them with
//! the break-even its model has measured (`DraftRoundCosts`): the smoothed
//! time of a round over the smoothed time of a plain token. Below it, the
//! rest of the reply is written plain. A reply cannot time the two against
//! each other itself: over a dozen tokens either reads anywhere from 15 to
//! 39 ms a token, more than the difference being decided.
//!
//! Until the model has timed plain tokens, a reply stops its rounds at the
//! first decision so that it does: one reply after a load gives up its
//! rounds' gain to learn the figure every later reply is decided by.

use std::time::Instant;

use crate::qwen3_5_streaming::model::draft_round_costs::DraftRoundCosts;

/// Reply tokens before a plain token's time counts, and rounds before a
/// round's does: the first of either after a prompt are slow while the
/// `astro_expert_cache` fills for the reply.
const SETTLING_REPLY_TOKEN_COUNT: u32 = 48;
const SETTLING_ROUND_COUNT: u32 = 8;
/// Rounds counted for each decision.
const ROUNDS_A_DECISION: u32 = 16;

/// What a reply's next forward should be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NextReplyStep {
    Round,
    /// Rounds decided too few tokens: this and every later token is plain.
    PlainToken,
}

/// One reply's count of what its rounds decide.
#[derive(Debug, Default)]
pub(super) struct DraftRoundPayoff {
    writes_plain_tokens: bool,
    reply_token_count: u32,
    round_count: u32,
    rounds_since_the_last_decision: u32,
    tokens_since_the_last_decision: u32,
    step_started_at: Option<Instant>,
}

impl DraftRoundPayoff {
    /// Whether the reply has gone back to plain tokens for good.
    pub(super) const fn writes_plain_tokens(&self) -> bool {
        self.writes_plain_tokens
    }

    /// Called where the reply is about to run a forward: closes the step
    /// before it, which decided `tokens_decided_by_the_last_step` reply
    /// tokens, and says what the next forward should be.
    pub(super) fn next_step(
        &mut self,
        now: Instant,
        tokens_decided_by_the_last_step: u32,
        costs: &DraftRoundCosts,
    ) -> NextReplyStep {
        self.close_the_last_step(now, tokens_decided_by_the_last_step, costs);
        if !self.writes_plain_tokens && self.rounds_since_the_last_decision >= ROUNDS_A_DECISION {
            let tokens_a_round = f64::from(self.tokens_since_the_last_decision)
                / f64::from(self.rounds_since_the_last_decision);
            let rounds_pay = costs
                .break_even_tokens_a_round()
                .is_some_and(|break_even| tokens_a_round > break_even);
            self.writes_plain_tokens = !rounds_pay;
            self.rounds_since_the_last_decision = 0;
            self.tokens_since_the_last_decision = 0;
        }
        self.step_started_at = Some(now);
        if self.writes_plain_tokens {
            NextReplyStep::PlainToken
        } else {
            NextReplyStep::Round
        }
    }

    fn close_the_last_step(
        &mut self,
        now: Instant,
        decided_token_count: u32,
        costs: &DraftRoundCosts,
    ) {
        let Some(step_started_at) = self.step_started_at.take() else {
            return;
        };
        let elapsed = now.saturating_duration_since(step_started_at);
        self.reply_token_count += decided_token_count;
        if self.writes_plain_tokens {
            if self.reply_token_count > SETTLING_REPLY_TOKEN_COUNT {
                costs.record_plain_token(elapsed);
            }
            return;
        }
        self.round_count += 1;
        if self.round_count > SETTLING_ROUND_COUNT {
            costs.record_round(elapsed);
            self.rounds_since_the_last_decision += 1;
            self.tokens_since_the_last_decision += decided_token_count;
        }
    }
}

#[cfg(test)]
#[path = "draft_round_payoff_tests.rs"]
mod tests;
