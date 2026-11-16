//! Unit tests for comparing the written model's answers with the trained ones.

use super::*;

#[test]
fn should_measure_gaps_and_never_pass_a_length_mismatch_or_a_non_finite_logit() {
    let close = largest_logit_gap(&[1.0, 100.0], &[1.001, 100.05]);
    assert!(close.within_tolerance);
    assert!((close.gap - 0.05).abs() < 1e-4);
    let apart = largest_logit_gap(&[1.0, 2.0], &[1.0, 2.1]);
    assert!(!apart.within_tolerance);
    assert!(!largest_logit_gap(&[1.0], &[1.0, 2.0]).within_tolerance);
    assert!(!largest_logit_gap(&[f32::NAN], &[1.0]).within_tolerance);
    assert!(largest_logit_gap(&[], &[]).within_tolerance);
}
