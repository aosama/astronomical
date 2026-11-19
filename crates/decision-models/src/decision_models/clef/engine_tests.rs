//! Unit tests for the engine's host arithmetic.

use super::plain_softmax;

#[test]
fn should_turn_logits_into_probabilities_without_a_temperature() {
    let probabilities = plain_softmax(&[0.0, 2.0_f64.ln()]);

    assert!((probabilities[0] - 1.0 / 3.0).abs() < 1e-12);
    assert!((probabilities[1] - 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn should_stay_finite_for_very_large_logits() {
    let probabilities = plain_softmax(&[1.0e6, 1.0e6 - 1.0]);

    assert!(
        probabilities
            .iter()
            .all(|probability| probability.is_finite())
    );
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn should_give_a_single_option_certainty() {
    assert_eq!(plain_softmax(&[-3.5]), [1.0]);
}
