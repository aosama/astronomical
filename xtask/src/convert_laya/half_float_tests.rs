//! Unit contracts for half-precision conversion: exact decoding, nearest
//! encoding with ties to even, and every boundary between number classes.

use super::{f32_to_half_bits, half_bits_to_f32};

#[test]
fn should_decode_common_values_exactly() {
    assert_eq!(half_bits_to_f32(0x0000), 0.0);
    assert_eq!(half_bits_to_f32(0x3C00), 1.0);
    assert_eq!(half_bits_to_f32(0xC000), -2.0);
    assert_eq!(half_bits_to_f32(0x3555), 0.333_251_95);
    assert_eq!(half_bits_to_f32(0x7BFF), 65_504.0);
}

#[test]
fn should_decode_subnormals_infinities_and_signed_zero() {
    assert_eq!(half_bits_to_f32(0x0001), 5.960_464_5e-8);
    assert_eq!(half_bits_to_f32(0x03FF), 1_023.0 * 5.960_464_5e-8);
    assert_eq!(half_bits_to_f32(0x7C00), f32::INFINITY);
    assert_eq!(half_bits_to_f32(0xFC00), f32::NEG_INFINITY);
    assert!(half_bits_to_f32(0x7E00).is_nan());
    assert!(half_bits_to_f32(0x8000).is_sign_negative());
}

#[test]
fn should_round_trip_every_finite_half() {
    for half_bits in 0..=u16::MAX {
        let decoded_value = half_bits_to_f32(half_bits);
        if decoded_value.is_nan() {
            continue;
        }
        assert_eq!(
            f32_to_half_bits(decoded_value),
            half_bits,
            "half bits {half_bits:#06x} must survive a round trip"
        );
    }
}

#[test]
fn should_round_to_the_nearest_half_with_ties_to_even() {
    // Between 1.0 (0x3C00) and the next half (0x3C01) lies 1 + 2^-11.
    let half_way_above_one = 1.0 + 1.0 / 2_048.0;
    assert_eq!(
        f32_to_half_bits(half_way_above_one),
        0x3C00,
        "a tie rounds to the even neighbor"
    );
    assert_eq!(
        f32_to_half_bits(f32::from_bits(half_way_above_one.to_bits() + 1)),
        0x3C01
    );
    // Between 0x3C01 and 0x3C02 the even neighbor is the upper one.
    let half_way_above_next = 1.0 + 3.0 / 2_048.0;
    assert_eq!(f32_to_half_bits(half_way_above_next), 0x3C02);
}

#[test]
fn should_carry_a_rounded_fraction_into_the_exponent() {
    // Just below 2.0: the fraction rounds up past ten bits.
    assert_eq!(
        f32_to_half_bits(f32::from_bits(2.0_f32.to_bits() - 1)),
        0x4000
    );
}

#[test]
fn should_turn_values_beyond_the_largest_half_into_infinity() {
    assert_eq!(f32_to_half_bits(65_504.0), 0x7BFF);
    assert_eq!(f32_to_half_bits(65_536.0), 0x7C00);
    assert_eq!(f32_to_half_bits(-1.0e30), 0xFC00);
    assert_eq!(f32_to_half_bits(f32::INFINITY), 0x7C00);
    // The largest half's own rounding boundary carries to infinity too.
    assert_eq!(f32_to_half_bits(65_520.0), 0x7C00);
}

#[test]
fn should_encode_tiny_values_as_subnormals_or_zero() {
    assert_eq!(f32_to_half_bits(5.960_464_5e-8), 0x0001);
    assert_eq!(f32_to_half_bits(1.0e-10), 0x0000);
    assert_eq!(f32_to_half_bits(-1.0e-10), 0x8000);
    // The largest subnormal rounds up into the smallest normal half.
    assert_eq!(f32_to_half_bits(6.103_515_6e-5), 0x0400);
}

#[test]
fn should_keep_not_a_number_as_not_a_number() {
    assert!(half_bits_to_f32(f32_to_half_bits(f32::NAN)).is_nan());
}
