use std::time::{Duration, Instant};

use super::{
    DraftRoundPayoff, NextReplyStep, ROUNDS_A_DECISION, SETTLING_REPLY_TOKEN_COUNT,
    SETTLING_ROUND_COUNT,
};
use crate::qwen3_5_streaming::model::draft_round_costs::DraftRoundCosts;

const ROUND: Duration = Duration::from_millis(38);
const PLAIN_TOKEN: Duration = Duration::from_millis(18);

/// A model that has timed both: 38 ms a round over 18 ms a token, so a
/// round breaks even at 2.11 tokens.
fn measured_costs() -> DraftRoundCosts {
    let costs = DraftRoundCosts::default();
    costs.record_round(ROUND);
    costs.record_plain_token(PLAIN_TOKEN);
    costs
}

/// Runs `round_count` rounds of `tokens_a_round` and returns the step
/// asked for after them.
fn step_after_rounds(
    payoff: &mut DraftRoundPayoff,
    costs: &DraftRoundCosts,
    round_count: u32,
    tokens_a_round: u32,
) -> NextReplyStep {
    let mut now = Instant::now();
    let mut next_step = payoff.next_step(now, 0, costs);
    for _ in 0..round_count {
        assert_eq!(next_step, NextReplyStep::Round);
        now += ROUND;
        next_step = payoff.next_step(now, tokens_a_round, costs);
    }
    next_step
}

#[test]
fn should_keep_drafting_when_rounds_decide_more_than_the_break_even() {
    let mut payoff = DraftRoundPayoff::default();
    let next_step = step_after_rounds(
        &mut payoff,
        &measured_costs(),
        SETTLING_ROUND_COUNT + 3 * ROUNDS_A_DECISION,
        3,
    );
    assert_eq!(next_step, NextReplyStep::Round);
    assert!(!payoff.writes_plain_tokens());
}

#[test]
fn should_go_back_to_plain_tokens_when_rounds_decide_fewer_than_the_break_even() {
    let mut payoff = DraftRoundPayoff::default();
    let next_step = step_after_rounds(
        &mut payoff,
        &measured_costs(),
        SETTLING_ROUND_COUNT + ROUNDS_A_DECISION,
        2,
    );
    assert_eq!(next_step, NextReplyStep::PlainToken);
    assert!(payoff.writes_plain_tokens());
}

#[test]
fn should_not_decide_before_the_settling_rounds_and_a_full_count() {
    let mut payoff = DraftRoundPayoff::default();
    let next_step = step_after_rounds(
        &mut payoff,
        &measured_costs(),
        SETTLING_ROUND_COUNT + ROUNDS_A_DECISION - 1,
        1,
    );
    assert_eq!(next_step, NextReplyStep::Round);
}

#[test]
fn should_stop_its_rounds_to_time_plain_tokens_when_the_model_has_timed_none() {
    let costs = DraftRoundCosts::default();
    let mut payoff = DraftRoundPayoff::default();
    let next_step = step_after_rounds(
        &mut payoff,
        &costs,
        SETTLING_ROUND_COUNT + ROUNDS_A_DECISION,
        4,
    );
    assert_eq!(next_step, NextReplyStep::PlainToken);
    assert_eq!(costs.break_even_tokens_a_round(), None);
    // The plain tokens after it are timed once the reply has settled.
    let mut now = Instant::now();
    for _ in 0..=SETTLING_REPLY_TOKEN_COUNT {
        now += PLAIN_TOKEN;
        assert_eq!(payoff.next_step(now, 1, &costs), NextReplyStep::PlainToken);
    }
    assert!(costs.break_even_tokens_a_round().is_some());
}

#[test]
fn should_never_return_to_rounds_once_it_writes_plain_tokens() {
    let costs = measured_costs();
    let mut payoff = DraftRoundPayoff::default();
    step_after_rounds(
        &mut payoff,
        &costs,
        SETTLING_ROUND_COUNT + ROUNDS_A_DECISION,
        1,
    );
    let mut now = Instant::now();
    for _ in 0..(4 * ROUNDS_A_DECISION) {
        now += PLAIN_TOKEN;
        assert_eq!(payoff.next_step(now, 1, &costs), NextReplyStep::PlainToken);
    }
}
