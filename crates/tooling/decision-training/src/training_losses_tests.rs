//! Unit tests for the training losses.

use super::*;

fn finite_difference_gradient(loss_of: fn(&[f32]) -> f64, option_logits: &[f32]) -> Vec<f64> {
    let step = 1e-3_f32;
    let mut gradient = Vec::with_capacity(option_logits.len());
    for option_index in 0..option_logits.len() {
        let mut raised = option_logits.to_vec();
        raised[option_index] += step;
        let mut lowered = option_logits.to_vec();
        lowered[option_index] -= step;
        gradient.push((loss_of(&raised) - loss_of(&lowered)) / (2.0 * f64::from(step)));
    }
    gradient
}

fn smoothed_loss_at_two(option_logits: &[f32]) -> f64 {
    smoothed_cross_entropy(option_logits, 2.0, 1).0
}

fn agreement_loss_at_two(option_logits: &[f32]) -> f64 {
    shipped_agreement_loss(option_logits, 2.0, &[0.2, 0.5, 0.3]).0
}

#[test]
fn should_give_the_smoothed_gradient_the_finite_differences_give() {
    let option_logits = [0.4_f32, -1.1, 2.3];
    let (_, analytic) = smoothed_cross_entropy(&option_logits, 2.0, 1);
    let numeric = finite_difference_gradient(smoothed_loss_at_two, &option_logits);
    for (analytic_value, numeric_value) in analytic.iter().zip(&numeric) {
        assert!((f64::from(*analytic_value) - numeric_value).abs() < 1e-4);
    }
}

#[test]
fn should_give_the_agreement_gradient_the_finite_differences_give() {
    let option_logits = [0.4_f32, -1.1, 2.3];
    let (_, analytic) = shipped_agreement_loss(&option_logits, 2.0, &[0.2, 0.5, 0.3]);
    let numeric = finite_difference_gradient(agreement_loss_at_two, &option_logits);
    for (analytic_value, numeric_value) in analytic.iter().zip(&numeric) {
        assert!((f64::from(*analytic_value) - numeric_value).abs() < 1e-4);
    }
}

#[test]
fn should_cost_nothing_to_agree_with_the_shipped_model_exactly() {
    let option_logits = [0.3_f32, 1.7, -0.4];
    let shipped_probabilities = tempered_probabilities(&option_logits, 1.5);
    let (loss, logit_gradients) =
        shipped_agreement_loss(&option_logits, 1.5, &shipped_probabilities);
    assert!(loss.abs() < 1e-12);
    for logit_gradient in logit_gradients {
        assert!(logit_gradient.abs() < 1e-7);
    }
}

#[test]
fn should_stop_pulling_once_the_right_answer_reaches_its_smoothed_target() {
    // With two options the smoothed target for the right one is 0.975: a
    // model already that sure feels no push to become surer.
    let target_logit_gap = (0.975_f64 / 0.025).ln() as f32;
    let (_, logit_gradients) = smoothed_cross_entropy(&[0.0, target_logit_gap], 1.0, 1);
    assert!(logit_gradients[1].abs() < 1e-6);
    let (_, overconfident) = smoothed_cross_entropy(&[0.0, 20.0], 1.0, 1);
    assert!(
        overconfident[1] > 0.0,
        "an overconfident answer is pulled back"
    );
}

#[test]
fn should_stay_finite_for_extreme_logits() {
    assert!(negative_log_likelihood(&[1000.0, -1000.0], 1.0, 1).is_finite());
    let (loss, _) = smoothed_cross_entropy(&[1000.0, -1000.0], 0.5, 1);
    assert!(loss.is_finite());
    assert!((negative_log_likelihood(&[0.0, 0.0], 1.0, 0) - 2.0_f64.ln()).abs() < 1e-12);
}
