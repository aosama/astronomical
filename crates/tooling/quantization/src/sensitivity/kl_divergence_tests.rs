//! Unit tests for the float32 KL reference. Parity with the Python formula
//! lives in `tests/integration/kl_divergence_parity.rs`.

use super::*;

#[test]
fn should_measure_zero_divergence_for_identical_logits() {
    let logits = [1.0_f32, 2.0, 3.0, 0.5, -1.0, 4.0];

    let kl_divergence = kl_divergence_reference_vs_probe(&logits, &logits, 3);

    assert!(kl_divergence.abs() < 1e-6);
}

#[test]
fn should_measure_a_positive_divergence_when_the_probe_moves_probability() {
    let reference_logits = [4.0_f32, 0.0, 0.0];
    let probe_logits = [0.0_f32, 4.0, 0.0];

    let kl_divergence = kl_divergence_reference_vs_probe(&reference_logits, &probe_logits, 3);

    assert!(kl_divergence > 1.0);
}

#[test]
fn should_stay_finite_for_logits_large_enough_to_overflow_a_plain_exponential() {
    let reference_logits = [1000.0_f32, 0.0];
    let probe_logits = [999.0_f32, 0.0];

    let kl_divergence = kl_divergence_reference_vs_probe(&reference_logits, &probe_logits, 2);

    assert!(kl_divergence.is_finite());
}

#[test]
fn should_report_nan_when_the_logit_lengths_disagree() {
    let kl_divergence = kl_divergence_reference_vs_probe(&[0.0, 1.0], &[0.0], 2);

    assert!(kl_divergence.is_nan());
}

#[test]
fn should_report_nan_for_empty_logits_or_a_zero_vocabulary() {
    assert!(kl_divergence_reference_vs_probe(&[], &[], 4).is_nan());
    assert!(kl_divergence_reference_vs_probe(&[0.0], &[0.0], 0).is_nan());
}
