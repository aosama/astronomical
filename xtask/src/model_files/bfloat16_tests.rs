//! Unit contracts for bfloat16 conversion: exact widening, nearest
//! narrowing with ties to even, and the special values.

use super::{bfloat16_bits_to_f32, f32_to_bfloat16_bits};

#[test]
fn should_widen_common_values_exactly() {
    assert_eq!(bfloat16_bits_to_f32(0x3F80), 1.0);
    assert_eq!(bfloat16_bits_to_f32(0xC000), -2.0);
    assert_eq!(bfloat16_bits_to_f32(0x0000), 0.0);
    assert_eq!(bfloat16_bits_to_f32(0x7F80), f32::INFINITY);
}

#[test]
fn should_round_trip_every_finite_bfloat16() {
    for bfloat16_bits in 0..=u16::MAX {
        let widened_value = bfloat16_bits_to_f32(bfloat16_bits);
        if widened_value.is_nan() {
            continue;
        }
        assert_eq!(f32_to_bfloat16_bits(widened_value), bfloat16_bits);
    }
}

#[test]
fn should_round_to_the_nearest_bfloat16_with_ties_to_even() {
    // 1.0 is 0x3F80_0000; the next bfloat16 above is 0x3F81_0000.
    assert_eq!(
        f32_to_bfloat16_bits(f32::from_bits(0x3F80_8000)),
        0x3F80,
        "a tie goes to the even neighbor"
    );
    assert_eq!(
        f32_to_bfloat16_bits(f32::from_bits(0x3F81_8000)),
        0x3F82,
        "a tie goes to the even neighbor"
    );
    assert_eq!(f32_to_bfloat16_bits(f32::from_bits(0x3F80_8001)), 0x3F81);
    assert_eq!(f32_to_bfloat16_bits(f32::from_bits(0x3F80_7FFF)), 0x3F80);
}

#[test]
fn should_carry_into_the_exponent_and_overflow_to_infinity() {
    assert_eq!(f32_to_bfloat16_bits(f32::from_bits(0x3FFF_FFFF)), 0x4000);
    assert_eq!(f32_to_bfloat16_bits(f32::MAX), 0x7F80);
    assert_eq!(f32_to_bfloat16_bits(-f32::MAX), 0xFF80);
}

#[test]
fn should_keep_not_a_number_and_signed_zero() {
    assert!(bfloat16_bits_to_f32(f32_to_bfloat16_bits(f32::NAN)).is_nan());
    assert_eq!(f32_to_bfloat16_bits(-0.0), 0x8000);
}
