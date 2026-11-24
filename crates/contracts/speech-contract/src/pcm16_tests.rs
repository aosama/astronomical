use super::pcm16_bytes;

#[test]
fn should_scale_to_sixteen_bit_little_endian() {
    assert_eq!(pcm16_bytes(&[0.0, 0.5]), [0, 0, 0x00, 0x40]);
}

// Boundaries: full scale and beyond clamp to the extremes.
#[test]
fn should_clamp_samples_beyond_full_scale() {
    let clamped = pcm16_bytes(&[1.0, 1.7, -1.0, -3.0]);
    assert_eq!(clamped, [0xff, 0x7f, 0xff, 0x7f, 0x01, 0x80, 0x01, 0x80]);
}

// Negative: a sample that is not a number becomes silence.
#[test]
fn should_silence_samples_that_are_not_numbers() {
    assert_eq!(pcm16_bytes(&[f32::NAN, f32::INFINITY]), [0, 0, 0, 0]);
}

#[test]
fn should_encode_no_samples_as_no_bytes() {
    assert!(pcm16_bytes(&[]).is_empty());
}
