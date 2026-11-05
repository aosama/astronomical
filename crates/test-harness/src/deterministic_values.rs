//! Reproducible bell-shaped values for operation-level measurements.
//!
//! A bare matrix multiply has no tokens, so the Romeo and Juliet fixture has
//! nothing to feed it until the projection is wired to a real model layer
//! (issue #161, rung 2). Until then, activations are unit-variance like the
//! output of RMSNorm and weights have the small spread of a trained
//! projection, which keeps the fp16 magnitudes representative.

/// The standard deviation of a trained 4,096-wide projection's weights.
pub const PROJECTION_WEIGHT_STANDARD_DEVIATION: f32 = 0.02;

/// SplitMix64: tiny, fast, and statistically sound enough for test inputs.
fn next_split_mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    mixed ^ (mixed >> 31)
}

fn next_unit_interval(state: &mut u64) -> f32 {
    // The top 24 bits fill an f32 mantissa exactly.
    (next_split_mix(state) >> 40) as f32 / (1u64 << 24) as f32
}

/// `value_count` values with mean 0 and the given standard deviation.
///
/// The sum of four uniform draws (Irwin–Hall) approximates a normal
/// distribution closely enough for magnitude realism, without the
/// transcendental calls of Box–Muller.
#[must_use]
pub fn bell_shaped_values(seed: u64, value_count: usize, standard_deviation: f32) -> Vec<f32> {
    let mut generator_state = seed;
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        let mut uniform_sum = 0.0_f32;
        for _ in 0..4 {
            uniform_sum += next_unit_interval(&mut generator_state);
        }
        // Four uniforms have mean 2 and variance 1/3; recentre and rescale
        // to unit variance before applying the requested spread.
        values.push((uniform_sum - 2.0) * 3f32.sqrt() * standard_deviation);
    }
    values
}

#[cfg(test)]
#[path = "deterministic_values_tests.rs"]
mod tests;
