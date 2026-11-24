//! Unit tests for the dense operations.

use super::*;

/// A 2 × 3 weight: rows [1, 2, 3] and [-1, 0, 4].
const WEIGHT: [f32; 6] = [1.0, 2.0, 3.0, -1.0, 0.0, 4.0];

#[test]
fn should_compute_a_linear_layer_row_by_row() {
    let mut output = vec![99.0];
    linear_forward(&WEIGHT, &[0.5, -0.5], &[1.0, 1.0, 2.0], &mut output);
    assert_eq!(output, vec![9.5, 6.5]);
}

#[test]
fn should_accumulate_parameter_gradients_as_outer_products() {
    let mut weight_gradient = vec![0.0; 6];
    let mut bias_gradient = vec![1.0, 1.0];
    linear_parameter_gradients(
        &[1.0, 2.0, 3.0],
        &[2.0, 0.0],
        &mut weight_gradient,
        &mut bias_gradient,
    );
    assert_eq!(weight_gradient, vec![2.0, 4.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(bias_gradient, vec![3.0, 1.0]);
}

#[test]
fn should_send_the_gradient_back_through_the_transposed_weight() {
    let mut input_gradient = vec![7.0];
    linear_input_gradient(&WEIGHT, &[1.0, 2.0], 3, &mut input_gradient);
    assert_eq!(input_gradient, vec![-1.0, 2.0, 11.0]);
}

#[test]
fn should_give_zero_for_empty_inputs() {
    assert_eq!(dot(&[], &[]), 0.0);
    let mut output = Vec::new();
    linear_forward(&[], &[], &[1.0], &mut output);
    assert!(output.is_empty());
}
