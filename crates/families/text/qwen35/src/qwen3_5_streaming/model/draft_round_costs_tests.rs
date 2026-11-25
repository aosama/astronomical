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
