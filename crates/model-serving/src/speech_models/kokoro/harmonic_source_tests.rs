use super::{SAMPLES_PER_CURVE_STEP, linear_upsampling_plan};

#[test]
fn should_place_each_sample_between_two_curve_steps_with_half_step_centres() {
    let upsampling_plan = linear_upsampling_plan(3);
    let (lower, upper, weights) = (
        &upsampling_plan.lower_steps,
        &upsampling_plan.upper_steps,
        &upsampling_plan.upper_weights,
    );
    assert_eq!(lower.len(), 3 * SAMPLES_PER_CURVE_STEP);
    // Sample 450 sits at source position (450.5 / 300) - 0.5, past step 1.
    assert_eq!((lower[450], upper[450]), (1, 2));
    assert!((weights[450] - (450.5 / 300.0 - 1.5) as f32).abs() < 1e-6);
}

// Boundary: the first half step clamps to step 0, the last to the final step.
#[test]
fn should_clamp_at_both_ends_of_the_curve() {
    let upsampling_plan = linear_upsampling_plan(2);
    let (lower, upper, weights) = (
        &upsampling_plan.lower_steps,
        &upsampling_plan.upper_steps,
        &upsampling_plan.upper_weights,
    );
    assert_eq!((lower[0], upper[0], weights[0]), (0, 1, 0.0));
    let last = lower.len() - 1;
    assert_eq!((lower[last], upper[last]), (1, 1));
}

// Negative: an empty curve plans no samples.
#[test]
fn should_plan_nothing_for_an_empty_curve() {
    let upsampling_plan = linear_upsampling_plan(0);
    assert!(upsampling_plan.lower_steps.is_empty() && upsampling_plan.upper_steps.is_empty());
    assert!(upsampling_plan.upper_weights.is_empty());
}
