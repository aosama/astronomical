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
