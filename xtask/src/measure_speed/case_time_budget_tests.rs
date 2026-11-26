use super::another_round_fits;

#[test]
fn should_run_another_round_while_it_fits_two_minutes() {
    assert!(another_round_fits(40.0, 30.0));
}

#[test]
fn should_fit_a_round_that_ends_exactly_on_the_budget_and_not_one_past_it() {
    assert!(another_round_fits(90.0, 30.0));
    assert!(!another_round_fits(90.1, 30.0));
}

#[test]
fn should_stop_after_one_round_on_a_model_whose_round_takes_over_a_minute() {
    assert!(!another_round_fits(75.0, 65.0));
}
