//! Unit tests for the log-softmax at a temperature.

use astronomical_decision_contract::calibrated_probabilities;

use super::*;
use crate::test_support::assert_close;

#[test]
fn should_match_the_probabilities_engines_serve() {
    // The serving engines use the contract's softmax; a fit is only valid if
    // this crate computes the very same numbers.
    let option_logits = [1.5_f32, -0.25, 3.0, 0.0];
    for temperature in [0.5, 1.0, 1.75, 5.0] {
        let served_probabilities = calibrated_probabilities(&option_logits, temperature);
        let computed_probabilities = probabilities_at(&option_logits, temperature);
        for (served, computed) in served_probabilities.iter().zip(&computed_probabilities) {
            assert_close(*computed, *served, 1e-12);
        }
    }
}

#[test]
fn should_sum_to_one() {
    let probabilities = probabilities_at(&[0.1, 0.2, 0.3, 4.0, -2.0], 1.3);
    assert_close(probabilities.iter().sum::<f64>(), 1.0, 1e-12);
}

#[test]
fn should_give_finite_log_probabilities_where_a_plain_probability_underflows() {
    let log_probabilities = log_probabilities_at(&[0.0, 2_000.0], 1.0);
    assert!(log_probabilities[0].is_finite());
    assert_close(log_probabilities[0], -2_000.0, 1e-9);
    assert_close(log_probabilities[1], 0.0, 1e-12);
}

#[test]
fn should_give_a_single_option_all_the_probability() {
    assert_eq!(probabilities_at(&[7.0], 2.0), vec![1.0]);
}

#[test]
fn should_pick_the_first_of_tied_largest_values() {
    assert_eq!(first_largest_index(&[0.2, 0.4, 0.4]), 1);
    assert_eq!(first_largest_index(&[0.9]), 0);
}
