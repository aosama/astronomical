//! Conversion between IEEE 754 half-precision bits and `f32`.
//!
//! Laya checkpoints store their weights as 16-bit floats. The converter
//! does its arithmetic in `f32`, so it decodes each value on the way in and
//! encodes scales and biases on the way out. `xtask` deliberately has no
//! numeric dependencies, and these two functions are all it needs.

const HALF_SIGN_MASK: u16 = 0x8000;
const HALF_EXPONENT_MASK: u16 = 0x7C00;
const HALF_FRACTION_MASK: u16 = 0x03FF;
const HALF_FRACTION_BITS: u32 = 10;
/// A half's exponent is stored with 15 added; a single's with 127 added.
const SINGLE_MINUS_HALF_EXPONENT_BIAS: u32 = 127 - 15;
const SINGLE_FRACTION_BITS: u32 = 23;
/// The smallest positive subnormal half is two to the power of minus 24.
const SMALLEST_HALF_SUBNORMAL: f32 = 5.960_464_5e-8;

/// The `f32` equal to the half-precision value with these bits. Every half
/// is exactly representable as a single, so nothing is rounded.
pub(crate) fn half_bits_to_f32(half_bits: u16) -> f32 {
    let sign_bit = u32::from(half_bits & HALF_SIGN_MASK) << 16;
    let stored_exponent = u32::from(half_bits & HALF_EXPONENT_MASK) >> HALF_FRACTION_BITS;
    let fraction = u32::from(half_bits & HALF_FRACTION_MASK);
    if stored_exponent == 0 {
        // Zero and subnormals: the fraction counts multiples of the smallest
        // subnormal, which an `f32` multiplies out exactly.
        let magnitude = fraction as f32 * SMALLEST_HALF_SUBNORMAL;
        return if sign_bit == 0 { magnitude } else { -magnitude };
    }
    let single_exponent = if stored_exponent == 0x1F {
        // Infinity and not-a-number keep an all-ones exponent.
        0xFF
    } else {
        stored_exponent + SINGLE_MINUS_HALF_EXPONENT_BIAS
    };
    f32::from_bits(
        sign_bit
            | (single_exponent << SINGLE_FRACTION_BITS)
            | (fraction << (SINGLE_FRACTION_BITS - HALF_FRACTION_BITS)),
    )
}

/// The bits of the half-precision value nearest to `single_value`, with
/// ties rounded to the even neighbor as IEEE 754 arithmetic does. A value
/// beyond the largest half becomes infinity.
pub(crate) fn f32_to_half_bits(single_value: f32) -> u16 {
    let single_bits = single_value.to_bits();
    let sign_bit = ((single_bits >> 16) & u32::from(HALF_SIGN_MASK)) as u16;
    if single_value.is_nan() {
        return sign_bit | HALF_EXPONENT_MASK | 0x0200;
    }
    let magnitude = single_value.abs();
    // Scaling by two to the power of 24 turns "multiples of the smallest
    // subnormal" into whole numbers, so one rounding covers subnormals.
    if magnitude < f32::from_bits(0x3880_0000) {
        let subnormal_steps = (magnitude / SMALLEST_HALF_SUBNORMAL).round_ties_even();
        // 1,024 steps is exactly the smallest normal half, whose bits are
        // the same number, so the carry needs no special case.
        return sign_bit | subnormal_steps as u16;
    }
    let single_exponent = (single_bits >> SINGLE_FRACTION_BITS) & 0xFF;
    let single_fraction = single_bits & 0x007F_FFFF;
    let dropped_bit_count = SINGLE_FRACTION_BITS - HALF_FRACTION_BITS;
    let kept_fraction = single_fraction >> dropped_bit_count;
    let dropped_fraction = single_fraction & ((1 << dropped_bit_count) - 1);
    let half_way = 1 << (dropped_bit_count - 1);
    let rounds_up =
        dropped_fraction > half_way || (dropped_fraction == half_way && kept_fraction & 1 == 1);
    // Exponent and fraction are packed before the carry is added, so a
    // fraction that rounds up past its ten bits carries into the exponent,
    // and an exponent that carries past 30 lands on infinity's bits.
    let half_exponent = single_exponent as i32 - SINGLE_MINUS_HALF_EXPONENT_BIAS as i32;
    if half_exponent >= 0x1F {
        return sign_bit | HALF_EXPONENT_MASK;
    }
    let packed_magnitude =
        ((half_exponent as u32) << HALF_FRACTION_BITS) + kept_fraction + u32::from(rounds_up);
    sign_bit | packed_magnitude as u16
}

#[cfg(test)]
#[path = "half_float_tests.rs"]
mod tests;
