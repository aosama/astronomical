//! Beside-source unit contracts for the pure PNG structural-validation
//! helpers in `png_validation.rs`. These pin the security-relevant arithmetic
//! — decoded-size overflow and ceiling enforcement, and the exact-IHDR field
//! layout — with plain byte fixtures, no decoder and no allocation.

use super::png_validation::decoded_rgb_byte_count_within_limit;

#[test]
fn should_compute_decoded_rgb_bytes_as_three_per_pixel() {
    let decoded_rgb_byte_count = decoded_rgb_byte_count_within_limit(4, 3, 1_000)
        .expect("a small truecolor frame must fit any sane ceiling");

    assert_eq!(decoded_rgb_byte_count, 4 * 3 * 3);
}

#[test]
fn should_reject_a_frame_above_the_decoded_size_ceiling() {
    // 100 x 100 x 3 = 30,000 bytes against a 29,999 ceiling: one byte over.
    let decoded_rgb_byte_count = decoded_rgb_byte_count_within_limit(100, 100, 29_999);

    assert!(decoded_rgb_byte_count.is_err());
}

#[test]
fn should_accept_a_frame_exactly_at_the_decoded_size_ceiling() {
    let decoded_rgb_byte_count = decoded_rgb_byte_count_within_limit(100, 100, 30_000);

    assert_eq!(decoded_rgb_byte_count, Ok(30_000));
}

#[test]
fn should_treat_pixel_count_overflow_as_a_resource_limit() {
    // u32::MAX on both axes cannot be represented as a usize pixel count on
    // any supported platform; untrusted dimensions must fail closed.
    let decoded_rgb_byte_count =
        decoded_rgb_byte_count_within_limit(u32::MAX, u32::MAX, usize::MAX);

    assert!(decoded_rgb_byte_count.is_err());
}
