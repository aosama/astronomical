use std::time::Duration;

use super::DraftRoundCosts;

#[test]
fn should_have_no_break_even_until_both_costs_were_timed() {
    let costs = DraftRoundCosts::default();
    assert_eq!(costs.break_even_tokens_a_round(), None);
    costs.record_round(Duration::from_millis(38));
    assert_eq!(costs.break_even_tokens_a_round(), None);
    costs.record_plain_token(Duration::from_millis(19));
    assert_eq!(costs.break_even_tokens_a_round(), Some(2.0));
}

#[test]
fn should_have_no_break_even_for_a_plain_token_that_took_no_time() {
    let costs = DraftRoundCosts::default();
    costs.record_round(Duration::from_millis(38));
    costs.record_plain_token(Duration::ZERO);
    assert_eq!(costs.break_even_tokens_a_round(), None);
}

#[test]
fn should_move_a_tenth_of_the_way_to_each_new_step() {
    let costs = DraftRoundCosts::default();
    costs.record_plain_token(Duration::from_millis(20));
    costs.record_round(Duration::from_millis(40));
    costs.record_round(Duration::from_millis(60));
    let break_even = costs
        .break_even_tokens_a_round()
        .expect("both costs are timed");
    assert!(
        (break_even - 2.1).abs() < 1e-9,
        "42 ms a round over 20 ms a token, got {break_even}"
    );
}

/// 40 ms a round over 10 ms a token: a round breaks even at four tokens.
fn costs_breaking_even_at_four_tokens() -> DraftRoundCosts {
    let costs = DraftRoundCosts::default();
    costs.record_round(Duration::from_millis(40));
    costs.record_plain_token(Duration::from_millis(10));
    costs
}

#[test]
fn should_try_rounds_while_nothing_is_known_about_them() {
    assert!(DraftRoundCosts::default().should_a_reply_try_rounds());
    assert!(costs_breaking_even_at_four_tokens().should_a_reply_try_rounds());
}

#[test]
fn should_try_rounds_when_they_have_decided_about_the_break_even() {
    let costs = costs_breaking_even_at_four_tokens();
    costs.record_tokens_a_round(3.7);
    assert!(costs.should_a_reply_try_rounds());
}

#[test]
fn should_not_try_rounds_that_never_decided_near_the_break_even() {
    let costs = costs_breaking_even_at_four_tokens();
    costs.record_tokens_a_round(2.5);
    costs.record_tokens_a_round(1.9);
    assert!(!costs.should_a_reply_try_rounds());
}

#[test]
fn should_try_again_after_sixteen_replies_without_rounds() {
    let costs = costs_breaking_even_at_four_tokens();
    costs.record_tokens_a_round(2.5);
    let mut tries = Vec::new();
    for _ in 0..32 {
        tries.push(costs.should_a_reply_try_rounds());
    }
    assert_eq!(tries.iter().filter(|tried| **tried).count(), 2);
    assert!(tries[15] && tries[31]);
}
