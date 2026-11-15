//! Unit tests for the exact GELU.

use super::*;

fn wide_gelu(value: f64) -> f64 {
    0.5 * value * (1.0 + erf(value * FRAC_1_SQRT_2))
}

#[test]
fn should_match_published_error_function_values() {
    // Values from Abramowitz and Stegun, table 7.1.
    for (input, expected) in [
        (0.0, 0.0),
        (0.5, 0.520_499_877_8),
        (1.0, 0.842_700_792_9),
        (2.0, 0.995_322_265_0),
        (-1.0, -0.842_700_792_9),
        (4.0, 0.999_999_984_6),
    ] {
        assert!((erf(input) - expected).abs() < 2e-7, "erf({input})");
    }
}

#[test]
fn should_give_the_exact_gelu_not_the_tanh_approximation() {
    // gelu(1) = 0.5 · (1 + erf(1/√2)) = 0.841 344 746; the tanh form gives 0.841 191 990.
    assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
    assert!((gelu(-3.0) - (-0.004_049_71)).abs() < 1e-6);
    assert_eq!(gelu(0.0), 0.0);
}

#[test]
fn should_match_finite_differences_for_the_derivative() {
    for input in [-4.0_f32, -1.3, -0.2, 0.0, 0.7, 2.5, 6.0] {
        let step = 1e-3_f64;
        let wide_input = f64::from(input);
        let finite_difference =
            (wide_gelu(wide_input + step) - wide_gelu(wide_input - step)) / (2.0 * step);
        assert!(
            (f64::from(gelu_derivative(input)) - finite_difference).abs() < 1e-5,
            "at {input}"
        );
    }
}
