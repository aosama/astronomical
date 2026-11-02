use super::{f16_bits_slice_to_f32, f16_bits_to_f32, f32_slice_to_f16_bits, f32_to_f16_bits};

#[test]
fn should_convert_exactly_representable_values_both_ways() {
    for (single, half_bits) in [
        (0.0_f32, 0x0000_u16),
        (1.0, 0x3c00),
        (-2.5, 0xc100),
        (0.5, 0x3800),
        (65504.0, 0x7bff),
    ] {
        assert_eq!(f32_to_f16_bits(single), half_bits, "narrowing {single}");
        assert_eq!(
            f16_bits_to_f32(half_bits),
            single,
            "widening {half_bits:#06x}"
        );
    }
}

#[test]
fn should_keep_the_sign_of_negative_zero() {
    assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
    assert!(f16_bits_to_f32(0x8000).is_sign_negative());
}

#[test]
fn should_round_halfway_cases_to_the_even_neighbour() {
    // 1 + 2⁻¹¹ sits exactly between 1.0 (even) and 1 + 2⁻¹⁰ (odd).
    assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
    // 1 + 3·2⁻¹¹ sits between 1 + 2⁻¹⁰ (odd) and 1 + 2⁻⁹ (even).
    assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
}

#[test]
fn should_overflow_beyond_the_largest_half_to_infinity() {
    assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);
    assert_eq!(f32_to_f16_bits(-1.0e9), 0xfc00);
    assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
}

#[test]
fn should_round_the_largest_half_neighbourhood_down_below_the_overflow_threshold() {
    assert_eq!(f32_to_f16_bits(65519.0), 0x7bff);
}

#[test]
fn should_represent_subnormal_halves() {
    let smallest_subnormal = 2f32.powi(-24);

    assert_eq!(f32_to_f16_bits(smallest_subnormal), 0x0001);
    assert_eq!(f16_bits_to_f32(0x0001), smallest_subnormal);
    assert_eq!(f32_to_f16_bits(2f32.powi(-15)), 0x0200);
}

#[test]
fn should_flush_values_below_half_the_smallest_subnormal_to_zero() {
    assert_eq!(f32_to_f16_bits(2f32.powi(-26)), 0x0000);
    // Exactly half the smallest subnormal ties to the even neighbour, zero.
    assert_eq!(f32_to_f16_bits(2f32.powi(-25)), 0x0000);
}

#[test]
fn should_preserve_not_a_number() {
    assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
}

#[test]
fn should_convert_slices_element_by_element() {
    let halves = f32_slice_to_f16_bits(&[1.0, -2.5]);

    assert_eq!(halves, vec![0x3c00, 0xc100]);
    assert_eq!(f16_bits_slice_to_f32(&halves), vec![1.0, -2.5]);
    assert!(f32_slice_to_f16_bits(&[]).is_empty());
}
