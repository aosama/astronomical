use super::bell_shaped_values;

#[test]
fn should_reproduce_the_same_values_for_the_same_seed() {
    assert_eq!(
        bell_shaped_values(7, 64, 1.0),
        bell_shaped_values(7, 64, 1.0)
    );
}

#[test]
fn should_produce_different_values_for_different_seeds() {
    assert_ne!(
        bell_shaped_values(7, 64, 1.0),
        bell_shaped_values(8, 64, 1.0)
    );
}

#[test]
fn should_centre_on_zero_with_the_requested_spread() {
    let values = bell_shaped_values(11, 100_000, 0.02);

    let mut sum = 0.0_f64;
    let mut sum_of_squares = 0.0_f64;
    for value in &values {
        sum += f64::from(*value);
        sum_of_squares += f64::from(*value) * f64::from(*value);
    }
    let mean = sum / values.len() as f64;
    let standard_deviation = (sum_of_squares / values.len() as f64 - mean * mean).sqrt();
    assert!(mean.abs() < 0.001, "mean {mean}");
    assert!(
        (standard_deviation - 0.02).abs() < 0.001,
        "standard deviation {standard_deviation}"
    );
}

#[test]
fn should_stay_within_the_irwin_hall_support() {
    let largest_magnitude = 2.0 * 3f32.sqrt();

    for value in bell_shaped_values(3, 10_000, 1.0) {
        assert!(value.abs() <= largest_magnitude, "value {value}");
    }
}

#[test]
fn should_return_nothing_when_no_values_are_requested() {
    assert!(bell_shaped_values(1, 0, 1.0).is_empty());
}
