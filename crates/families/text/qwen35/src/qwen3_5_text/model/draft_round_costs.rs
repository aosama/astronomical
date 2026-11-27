//! What a plain reply token and a draft round cost this model, as it is
//! serving now: the two figures a reply needs to tell whether its rounds
//! pay (`inference_execution::draft_round_payoff`).
//!
//! They are kept on the model, across replies, because they depend on the
//! model and the memory it was given (how many experts a token reads from
//! storage), not on what a reply says. Each is smoothed over the steps
//! that report it, so one slow step does not move a decision.

use std::cell::Cell;
use std::time::Duration;

/// The weight of one new step in a smoothed cost: a cost settles over
/// about ten steps and follows a change in a few dozen.
const NEW_STEP_WEIGHT: f64 = 0.1;

/// How far above the most tokens a round has kept the break-even may sit
/// before rounds are not tried: counts of sixteen rounds vary by about a
/// tenth between replies of one kind.
const BREAK_EVEN_TOLERANCE: f64 = 1.1;
/// Replies written plain before rounds are tried again, in case what is
/// being written now keeps more than anything before it did.
const REPLIES_BETWEEN_NEW_TRIES: u32 = 16;

/// Smoothed seconds of a plain reply token and of a draft round, and the
/// most tokens a round its rounds have decided.
#[derive(Debug, Default)]
pub struct DraftRoundCosts {
    plain_token_seconds: Cell<Option<f64>>,
    round_seconds: Cell<Option<f64>>,
    most_tokens_a_round: Cell<Option<f64>>,
    replies_since_rounds_were_tried: Cell<u32>,
}

impl DraftRoundCosts {
    pub fn record_plain_token(&self, elapsed: Duration) {
        record(&self.plain_token_seconds, elapsed);
    }

    pub fn record_round(&self, elapsed: Duration) {
        record(&self.round_seconds, elapsed);
    }

    /// Records what one count of rounds decided, a round.
    pub fn record_tokens_a_round(&self, tokens_a_round: f64) {
        let most_so_far = self.most_tokens_a_round.get().unwrap_or(0.0);
        self.most_tokens_a_round
            .set(Some(most_so_far.max(tokens_a_round)));
    }

    /// Whether a new reply should try rounds. It should not when the
    /// break-even is beyond anything rounds have decided on this model:
    /// a dense 27B's tree pass in MLX's layout costs four tokens' time
    /// where its rounds keep two and a half, and trying sixteen rounds a
    /// reply to learn that again cost each reply about two seconds. Such a
    /// model still tries once every few replies.
    pub fn should_a_reply_try_rounds(&self) -> bool {
        let (Some(break_even), Some(most_tokens_a_round)) = (
            self.break_even_tokens_a_round(),
            self.most_tokens_a_round.get(),
        ) else {
            return true;
        };
        if most_tokens_a_round * BREAK_EVEN_TOLERANCE >= break_even {
            return true;
        }
        let replies_since_a_try = self.replies_since_rounds_were_tried.get() + 1;
        let tries_again = replies_since_a_try >= REPLIES_BETWEEN_NEW_TRIES;
        self.replies_since_rounds_were_tried
            .set(if tries_again { 0 } else { replies_since_a_try });
        // Without this line a reply that never drafts leaves no trace of
        // why; it fires once a reply, so it costs nothing to keep.
        tracing::info!(
            break_even_tokens_a_round = break_even,
            most_tokens_a_round,
            round_milliseconds = ?self.round_milliseconds(),
            plain_token_milliseconds = ?self.plain_token_milliseconds(),
            tries_again,
            "draft rounds have not paid on this model as it serves now"
        );
        tries_again
    }

    /// The smoothed time of a draft round, once one was timed.
    pub fn round_milliseconds(&self) -> Option<f64> {
        self.round_seconds.get().map(seconds_as_milliseconds)
    }

    /// The smoothed time of a plain reply token, once one was timed.
    pub fn plain_token_milliseconds(&self) -> Option<f64> {
        self.plain_token_seconds.get().map(seconds_as_milliseconds)
    }

    /// How many reply tokens a round must decide to cost what plain tokens
    /// cost; `None` until both a plain token and a round were timed.
    pub fn break_even_tokens_a_round(&self) -> Option<f64> {
        let plain_token_seconds = self.plain_token_seconds.get()?;
        let round_seconds = self.round_seconds.get()?;
        (plain_token_seconds > 0.0).then_some(round_seconds / plain_token_seconds)
    }
}

fn seconds_as_milliseconds(seconds: f64) -> f64 {
    seconds * 1_000.0
}

fn record(smoothed_seconds: &Cell<Option<f64>>, elapsed: Duration) {
    let step_seconds = elapsed.as_secs_f64();
    smoothed_seconds.set(Some(match smoothed_seconds.get() {
        Some(earlier_seconds) => {
            earlier_seconds + NEW_STEP_WEIGHT * (step_seconds - earlier_seconds)
        }
        None => step_seconds,
    }));
}

#[cfg(test)]
#[path = "draft_round_costs_tests.rs"]
mod tests;
