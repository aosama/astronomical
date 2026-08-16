//! Architecture-neutral rotary frequency denominators for MLX `fast.rope`.
//!
//! MLX takes the reciprocal of the supplied `freqs` array. This owner therefore
//! emits denominators `theta^(2i/d)`, never inverse frequencies. YaRN blends
//! these denominators while family owners retain their own attention factor.

use std::f64::consts::PI;

/// Validated YaRN frequency denominators ready for MLX `fast.rope`.
#[derive(Clone, Debug, PartialEq)]
pub struct YarnRopeFrequencyDenominators {
    frequency_denominators: Vec<f32>,
}

impl YarnRopeFrequencyDenominators {
    /// Returns the rank-one denominator vector of length `rotary_dimension / 2`.
    #[must_use]
    pub fn frequency_denominators(&self) -> &[f32] {
        &self.frequency_denominators
    }
}

/// Errors from rotary frequency-denominator construction.
#[derive(Clone, Debug, PartialEq)]
pub enum RopeFrequencyError {
    /// Rotary width was zero or odd.
    InvalidRotaryDimension {
        rotary_dimension: u32,
        description: &'static str,
    },
    /// Theta was not a finite value greater than one.
    InvalidTheta {
        theta: f64,
        description: &'static str,
    },
    /// The context-extension factor was not a positive finite value.
    InvalidFactor {
        factor: f64,
        description: &'static str,
    },
    /// The original training context was zero.
    InvalidOriginalMaximumPositionCount {
        original_maximum_position_count: u32,
        description: &'static str,
    },
    /// A YaRN rotation-count threshold was invalid.
    InvalidBeta { description: &'static str },
}

/// Builds default RoPE denominators `theta^(2i/d)` for `i` in `[0, d/2)`.
pub fn compute_default_rope_frequency_denominators(
    theta: f64,
    rotary_dimension: u32,
) -> Result<Vec<f32>, RopeFrequencyError> {
    validate_rotary_dimension(rotary_dimension)?;
    validate_theta(theta)?;
    let pair_count = rotary_dimension / 2;
    let mut frequency_denominators = Vec::with_capacity(pair_count as usize);
    for pair_index in 0..pair_count {
        frequency_denominators.push(default_frequency_denominator(
            theta,
            rotary_dimension,
            pair_index,
        ) as f32);
    }
    Ok(frequency_denominators)
}

/// Builds YaRN-blended frequency denominators for MLX `fast.rope`.
pub fn compute_yarn_rope_frequency_denominators(
    theta: f64,
    rotary_dimension: u32,
    original_maximum_position_count: u32,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Result<YarnRopeFrequencyDenominators, RopeFrequencyError> {
    validate_rotary_dimension(rotary_dimension)?;
    validate_theta(theta)?;
    if !factor.is_finite() || factor <= 0.0 {
        return Err(RopeFrequencyError::InvalidFactor {
            factor,
            description: "YaRN factor must be a positive finite value",
        });
    }
    if original_maximum_position_count == 0 {
        return Err(RopeFrequencyError::InvalidOriginalMaximumPositionCount {
            original_maximum_position_count,
            description: "original maximum position count must be positive",
        });
    }
    if !beta_fast.is_finite() || beta_fast <= 0.0 || !beta_slow.is_finite() || beta_slow <= 0.0 {
        return Err(RopeFrequencyError::InvalidBeta {
            description: "beta_fast and beta_slow must be positive finite rotation counts",
        });
    }
    if beta_fast < beta_slow {
        return Err(RopeFrequencyError::InvalidBeta {
            description: "beta_fast must be greater than or equal to beta_slow",
        });
    }

    let (ramp_low, ramp_high) = yarn_correction_range(
        rotary_dimension,
        original_maximum_position_count,
        theta,
        beta_fast,
        beta_slow,
    );
    let pair_count = rotary_dimension / 2;
    let mut frequency_denominators = Vec::with_capacity(pair_count as usize);
    for pair_index in 0..pair_count {
        let extra_denominator = default_frequency_denominator(theta, rotary_dimension, pair_index);
        let interpolated_denominator = factor * extra_denominator;
        let keep_unscaled_weight =
            1.0 - yarn_linear_ramp(f64::from(pair_index), ramp_low, ramp_high);
        // Harmonic blending preserves the published inverse-frequency ramp.
        let blended_denominator = interpolated_denominator * keep_unscaled_weight
            + extra_denominator * (1.0 - keep_unscaled_weight);
        frequency_denominators
            .push(((interpolated_denominator * extra_denominator) / blended_denominator) as f32);
    }
    Ok(YarnRopeFrequencyDenominators {
        frequency_denominators,
    })
}

fn default_frequency_denominator(theta: f64, rotary_dimension: u32, pair_index: u32) -> f64 {
    theta.powf(2.0 * f64::from(pair_index) / f64::from(rotary_dimension))
}

fn yarn_correction_range(
    rotary_dimension: u32,
    original_maximum_position_count: u32,
    theta: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> (f64, f64) {
    let low = yarn_correction_dimension(
        beta_fast,
        rotary_dimension,
        original_maximum_position_count,
        theta,
    )
    .floor()
    .max(0.0);
    let high = yarn_correction_dimension(
        beta_slow,
        rotary_dimension,
        original_maximum_position_count,
        theta,
    )
    .ceil()
    .min(f64::from(rotary_dimension.saturating_sub(1)));
    (low, high)
}

fn yarn_correction_dimension(
    rotation_count: f64,
    rotary_dimension: u32,
    original_maximum_position_count: u32,
    theta: f64,
) -> f64 {
    f64::from(rotary_dimension)
        * (f64::from(original_maximum_position_count) / (rotation_count * 2.0 * PI)).ln()
        / (2.0 * theta.ln())
}

fn yarn_linear_ramp(pair_index: f64, ramp_low: f64, ramp_high: f64) -> f64 {
    let mut effective_high = ramp_high;
    if (effective_high - ramp_low).abs() < f64::EPSILON {
        effective_high += 0.001;
    }
    ((pair_index - ramp_low) / (effective_high - ramp_low)).clamp(0.0, 1.0)
}

fn validate_rotary_dimension(rotary_dimension: u32) -> Result<(), RopeFrequencyError> {
    if rotary_dimension == 0 {
        return Err(RopeFrequencyError::InvalidRotaryDimension {
            rotary_dimension,
            description: "rotary dimension must be positive",
        });
    }
    if !rotary_dimension.is_multiple_of(2) {
        return Err(RopeFrequencyError::InvalidRotaryDimension {
            rotary_dimension,
            description: "rotary dimension must be even",
        });
    }
    Ok(())
}

fn validate_theta(theta: f64) -> Result<(), RopeFrequencyError> {
    if !theta.is_finite() || theta <= 1.0 {
        return Err(RopeFrequencyError::InvalidTheta {
            theta,
            description: "theta must be a finite value greater than one",
        });
    }
    Ok(())
}
