//! The exact (error-function) GELU and its derivative, as the served heads
//! compute it.
//!
//! Laya's scorer and Clef's residual scorer both use `0.5 · x · (1 + erf(x/√2))`,
//! not the tanh approximation; a CPU copy that used the approximation would
//! disagree with serving by up to about 1e-3 per activation and fail the
//! parity check before training starts. Rust's `f64::erf` is still unstable,
//! so the error function is computed here with the Chebyshev-fitted
//! complementary error function from Numerical Recipes (`erfcc`), whose
//! fractional error is below 1.2e-7 everywhere: far below the 1e-4 the parity
//! check allows.

const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1 / √(2π)`, the standard normal density's scale.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// The error function, to about 1.2e-7.
#[must_use]
pub fn erf(input: f64) -> f64 {
    1.0 - complementary_erf(input)
}

/// `erfc(x) = 1 − erf(x)`, Numerical Recipes' `erfcc`.
fn complementary_erf(input: f64) -> f64 {
    let magnitude = input.abs();
    let t = 1.0 / (1.0 + 0.5 * magnitude);
    let polynomial = -magnitude * magnitude - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let tail = t * polynomial.exp();
    if input >= 0.0 { tail } else { 2.0 - tail }
}

/// `0.5 · x · (1 + erf(x/√2))`.
#[must_use]
pub fn gelu(input: f32) -> f32 {
    let wide_input = f64::from(input);
    (0.5 * wide_input * (1.0 + erf(wide_input * FRAC_1_SQRT_2))) as f32
}

/// `d gelu / dx = Φ(x) + x · φ(x)`, the normal distribution and density.
#[must_use]
pub fn gelu_derivative(input: f32) -> f32 {
    let wide_input = f64::from(input);
    let cumulative = 0.5 * (1.0 + erf(wide_input * FRAC_1_SQRT_2));
    let density = FRAC_1_SQRT_2PI * (-0.5 * wide_input * wide_input).exp();
    (cumulative + wide_input * density) as f32
}

#[cfg(test)]
#[path = "gelu_tests.rs"]
mod tests;
