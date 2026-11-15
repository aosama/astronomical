//! Unit tests for the anchored Adam optimizer.

use super::*;

fn plain_settings() -> AdamSettings {
    AdamSettings {
        learning_rate: 0.1,
        anchor_decay: 0.0,
        maximum_gradient_norm: 1_000.0,
        ..HEAD_TRAINING_ADAM_SETTINGS
    }
}

#[test]
fn should_take_a_first_step_of_the_learning_rate_against_the_gradient() {
    // Bias-corrected Adam's first step is lr · g / (|g| + ε): one learning rate.
    let mut optimizer = AnchoredAdam::new(plain_settings(), 2);
    let mut parameters = vec![1.0, -1.0];
    let mut gradient = vec![3.0, -0.5];
    optimizer
        .step(&mut parameters, &mut gradient, &[1.0, -1.0])
        .unwrap();
    assert!((parameters[0] - 0.9).abs() < 1e-6);
    assert!((parameters[1] - (-0.9)).abs() < 1e-6);
}

#[test]
fn should_pull_toward_the_shipped_weights_not_toward_zero() {
    let settings = AdamSettings {
        anchor_decay: 0.5,
        ..plain_settings()
    };
    let mut optimizer = AnchoredAdam::new(settings, 1);
    let mut parameters = vec![3.0];
    // No gradient: only the anchor pull acts, toward 2, not toward 0.
    optimizer.step(&mut parameters, &mut [0.0], &[2.0]).unwrap();
    assert!((parameters[0] - (3.0 - 0.1 * 0.5 * 1.0)).abs() < 1e-6);
}

#[test]
fn should_shorten_a_gradient_longer_than_allowed() {
    let settings = AdamSettings {
        maximum_gradient_norm: 1.0,
        ..plain_settings()
    };
    let mut optimizer = AnchoredAdam::new(settings, 2);
    let mut gradient = vec![30.0, 40.0];
    optimizer
        .step(&mut [0.0, 0.0], &mut gradient, &[0.0, 0.0])
        .unwrap();
    assert!((gradient[0] - 0.6).abs() < 1e-6 && (gradient[1] - 0.8).abs() < 1e-6);
}

#[test]
fn should_refuse_a_non_finite_gradient_without_changing_anything() {
    let mut optimizer = AnchoredAdam::new(plain_settings(), 2);
    let mut parameters = vec![1.0, 2.0];
    optimizer
        .step(&mut parameters, &mut [0.5, 0.5], &[1.0, 2.0])
        .unwrap();
    let parameters_before = parameters.clone();
    let first_moment_before = optimizer.first_moment.clone();
    let refusal = optimizer.step(&mut parameters, &mut [f32::NAN, 0.5], &[1.0, 2.0]);
    assert_eq!(refusal, Err(StepRefusal::NonFiniteGradient));
    assert_eq!(parameters, parameters_before);
    assert_eq!(optimizer.first_moment, first_moment_before);
    assert_eq!(optimizer.step_count, 1);
    assert!(
        optimizer
            .step(&mut parameters, &mut [f32::INFINITY, 0.0], &[1.0, 2.0])
            .is_err()
    );
}
