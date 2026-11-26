//! Values for measurements that run on synthetic tensors.

/// Deterministic values in [-1, 1): a linear congruential stream, so a
/// measurement needs no random-number dependency and repeats exactly.
pub(super) fn pseudo_random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        values.push(((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0);
    }
    values
}

/// Values spread evenly over `-magnitude..magnitude`, from a xorshift stream
/// the caller carries from one array to the next so no two arrays of a
/// sweep repeat. A parity sweep draws at several magnitudes: arithmetic that
/// agrees near zero can part far from it, which patterned inputs never
/// reach (issue #2599).
pub(super) fn swept_values(count: usize, magnitude: f32, generator_state: &mut u64) -> Vec<f32> {
    let mut swept = Vec::with_capacity(count);
    for _ in 0..count {
        swept.push((next_unit_interval_value(generator_state) * 2.0 - 1.0) * magnitude);
    }
    swept
}

/// Values spread evenly over `0..1`, for rates and decays.
pub(super) fn swept_unit_interval_values(count: usize, generator_state: &mut u64) -> Vec<f32> {
    let mut swept = Vec::with_capacity(count);
    for _ in 0..count {
        swept.push(next_unit_interval_value(generator_state));
    }
    swept
}

fn next_unit_interval_value(generator_state: &mut u64) -> f32 {
    *generator_state ^= *generator_state << 13;
    *generator_state ^= *generator_state >> 7;
    *generator_state ^= *generator_state << 17;
    (*generator_state >> 40) as f32 / (1_u64 << 24) as f32
}

/// The largest difference between two results as a share of the larger
/// result's own largest magnitude, and how many elements differ in any bit.
pub(super) fn largest_relative_difference(actual: &[f32], expected: &[f32]) -> (f32, usize) {
    assert_eq!(actual.len(), expected.len());
    let mut largest_magnitude = f32::MIN_POSITIVE;
    let mut largest_difference = 0.0_f32;
    let mut differing_element_count = 0;
    for (actual_value, expected_value) in actual.iter().zip(expected) {
        largest_magnitude = largest_magnitude.max(expected_value.abs());
        largest_difference = largest_difference.max((actual_value - expected_value).abs());
        if actual_value.to_bits() != expected_value.to_bits() {
            differing_element_count += 1;
        }
    }
    (
        largest_difference / largest_magnitude,
        differing_element_count,
    )
}
