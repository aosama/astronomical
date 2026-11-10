//! Conversion between bfloat16 bits and `f32`.
//!
//! A bfloat16 value is the upper half of an `f32`: the same sign and
//! exponent, and the top seven fraction bits. Widening is therefore exact,
//! and narrowing is one rounding of the dropped low half.

/// The `f32` equal to the bfloat16 value with these bits.
pub(crate) fn bfloat16_bits_to_f32(bfloat16_bits: u16) -> f32 {
    f32::from_bits(u32::from(bfloat16_bits) << 16)
}

/// The bits of the bfloat16 value nearest to `single_value`, ties to even,
/// as PyTorch's `.to(torch.bfloat16)` rounds. Not-a-number stays
/// not-a-number with its sign.
pub(crate) fn f32_to_bfloat16_bits(single_value: f32) -> u16 {
    let single_bits = single_value.to_bits();
    if single_value.is_nan() {
        // Keep a fraction bit set, or the truncation could turn it into infinity.
        return ((single_bits >> 16) as u16) | 0x0040;
    }
    // Adding just under half of the dropped range, plus the kept part's
    // lowest bit, rounds to nearest with ties to even in integer arithmetic.
    // A carry out of the fraction lands in the exponent, which is correct,
    // and out of the largest finite value lands on infinity, also correct.
    let lowest_kept_bit = (single_bits >> 16) & 1;
    let rounded_bits = single_bits + 0x7FFF + lowest_kept_bit;
    (rounded_bits >> 16) as u16
}

#[cfg(test)]
#[path = "bfloat16_tests.rs"]
mod tests;
