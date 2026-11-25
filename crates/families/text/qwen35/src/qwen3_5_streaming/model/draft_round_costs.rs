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

/// Smoothed seconds of a plain reply token and of a draft round.
#[derive(Debug, Default)]
pub struct DraftRoundCosts {
    plain_token_seconds: Cell<Option<f64>>,
    round_seconds: Cell<Option<f64>>,
}

impl DraftRoundCosts {
    pub fn record_plain_token(&self, elapsed: Duration) {
        record(&self.plain_token_seconds, elapsed);
    }

    pub fn record_round(&self, elapsed: Duration) {
        record(&self.round_seconds, elapsed);
    }

    /// How many reply tokens a round must decide to cost what plain tokens
    /// cost; `None` until both a plain token and a round were timed.
    pub fn break_even_tokens_a_round(&self) -> Option<f64> {
        let plain_token_seconds = self.plain_token_seconds.get()?;
        let round_seconds = self.round_seconds.get()?;
        (plain_token_seconds > 0.0).then_some(round_seconds / plain_token_seconds)
    }
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
