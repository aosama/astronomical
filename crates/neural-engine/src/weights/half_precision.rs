//! IEEE 754 binary16 conversion for the fp16 surfaces the Neural Engine reads.
//!
//! Written out here instead of taking a crate dependency because the two
//! conversions are short, exhaustively testable, and the only half-precision
//! need this research crate has.

const F32_EXPONENT_BIAS: i32 = 127;
const F16_EXPONENT_BIAS: i32 = 15;
const F16_MAXIMUM_BIASED_EXPONENT: i32 = 31;
const F16_INFINITY_BITS: u16 = 0x7c00;
const F16_QUIET_NAN_BITS: u16 = 0x7e00;

/// Rounds an `f32` to the nearest binary16 value, ties to even, the same
/// rounding Core ML and Metal apply when they narrow to half precision.
#[must_use]
pub fn f32_to_f16_bits(single: f32) -> u16 {
    let single_bits = single.to_bits();
    let sign_bits = ((single_bits >> 16) & 0x8000) as u16;
    let biased_exponent = ((single_bits >> 23) & 0xff) as i32;
    let mantissa = single_bits & 0x007f_ffff;
    if biased_exponent == 0xff {
        let special_bits = if mantissa == 0 {
            F16_INFINITY_BITS
        } else {
            F16_QUIET_NAN_BITS
        };
        return sign_bits | special_bits;
    }
    let half_exponent = biased_exponent - F32_EXPONENT_BIAS + F16_EXPONENT_BIAS;
    if half_exponent >= F16_MAXIMUM_BIASED_EXPONENT {
        return sign_bits | F16_INFINITY_BITS;
    }
    if half_exponent <= 0 {
        return sign_bits | subnormal_half_magnitude(half_exponent, mantissa);
    }
    let truncated = ((half_exponent as u32) << 10) | (mantissa >> 13);
    // Rounding can carry into the exponent and up to infinity; both are the
    // correctly rounded binary16 results, so the carry is kept as is.
    sign_bits | round_to_nearest_even(truncated, mantissa & 0x1fff, 0x1000) as u16
}

fn subnormal_half_magnitude(half_exponent: i32, mantissa: u32) -> u16 {
    // Below 2⁻²⁵ even the largest single mantissa rounds to zero.
    if half_exponent < -10 {
        return 0;
    }
    let significand = mantissa | 0x0080_0000;
    let shift = (14 - half_exponent) as u32;
    let truncated = significand >> shift;
    let remainder = significand & ((1 << shift) - 1);
    round_to_nearest_even(truncated, remainder, 1 << (shift - 1)) as u16
}

fn round_to_nearest_even(truncated: u32, remainder: u32, halfway: u32) -> u32 {
    let rounds_up = remainder > halfway || (remainder == halfway && truncated & 1 == 1);
    truncated + u32::from(rounds_up)
}

/// Widens a binary16 value to `f32`; exact, because every half value is a
/// single value.
#[must_use]
pub fn f16_bits_to_f32(half: u16) -> f32 {
    let sign_bits = u32::from(half & 0x8000) << 16;
    let biased_exponent = i32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x03ff);
    if biased_exponent == F16_MAXIMUM_BIASED_EXPONENT {
        return f32::from_bits(sign_bits | 0x7f80_0000 | (mantissa << 13));
    }
    if biased_exponent == 0 {
        let magnitude = mantissa as f32 * 2f32.powi(-24);
        return if sign_bits == 0 {
            magnitude
        } else {
            -magnitude
        };
    }
    let single_exponent = (biased_exponent - F16_EXPONENT_BIAS + F32_EXPONENT_BIAS) as u32;
    f32::from_bits(sign_bits | (single_exponent << 23) | (mantissa << 13))
}

/// Narrows a slice to binary16 bit patterns.
#[must_use]
pub fn f32_slice_to_f16_bits(singles: &[f32]) -> Vec<u16> {
    let mut halves = Vec::with_capacity(singles.len());
    for single in singles {
        halves.push(f32_to_f16_bits(*single));
    }
    halves
}

/// Widens a slice of binary16 bit patterns.
#[must_use]
pub fn f16_bits_slice_to_f32(halves: &[u16]) -> Vec<f32> {
    let mut singles = Vec::with_capacity(halves.len());
    for half in halves {
        singles.push(f16_bits_to_f32(*half));
    }
    singles
}

#[cfg(test)]
#[path = "half_precision_tests.rs"]
mod tests;
