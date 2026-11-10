//! Unit contracts for affine quantization: every weight is recovered to
//! within half a step, zero is exact, and malformed input is refused.

use super::super::half_float::half_bits_to_f32;
use super::{QUANTIZATION_GROUP_SIZE, quantize_matrix};

/// What the engine computes for each weight: `level * scale + bias`.
fn dequantized_weights(weights: &[f32]) -> Vec<f32> {
    let quantized_matrix = quantize_matrix(weights).expect("the weights should quantize");
    let mut recovered_weights = Vec::with_capacity(weights.len());
    for (weight_index, quantized_level) in quantized_matrix.quantized_levels.iter().enumerate() {
        let group_index = weight_index / QUANTIZATION_GROUP_SIZE;
        let scale = half_bits_to_f32(quantized_matrix.scale_half_bits[group_index]);
        let bias = half_bits_to_f32(quantized_matrix.bias_half_bits[group_index]);
        recovered_weights.push(f32::from(*quantized_level) * scale + bias);
    }
    recovered_weights
}

fn evenly_spread_group(lowest_weight: f32, highest_weight: f32) -> Vec<f32> {
    let mut group_weights = Vec::with_capacity(QUANTIZATION_GROUP_SIZE);
    for weight_index in 0..QUANTIZATION_GROUP_SIZE {
        let fraction = weight_index as f32 / (QUANTIZATION_GROUP_SIZE - 1) as f32;
        group_weights.push(lowest_weight + fraction * (highest_weight - lowest_weight));
    }
    group_weights
}

fn assert_recovered_within_half_a_step(weights: &[f32]) {
    let quantized_matrix = quantize_matrix(weights).expect("the weights should quantize");
    let recovered_weights = dequantized_weights(weights);
    for (weight_index, (weight, recovered_weight)) in
        weights.iter().zip(&recovered_weights).enumerate()
    {
        let group_index = weight_index / QUANTIZATION_GROUP_SIZE;
        let step = half_bits_to_f32(quantized_matrix.scale_half_bits[group_index]).abs();
        assert!(
            (weight - recovered_weight).abs() <= step * 0.5 + f32::EPSILON,
            "weight {weight} was recovered as {recovered_weight}, more than half a step ({step}) away"
        );
    }
}

#[test]
fn should_recover_a_group_that_spans_zero_within_half_a_step() {
    assert_recovered_within_half_a_step(&evenly_spread_group(-0.25, 0.75));
}

#[test]
fn should_recover_groups_whose_larger_extreme_is_negative_or_positive() {
    assert_recovered_within_half_a_step(&evenly_spread_group(-0.9, 0.1));
    assert_recovered_within_half_a_step(&evenly_spread_group(0.02, 0.4));
    assert_recovered_within_half_a_step(&evenly_spread_group(-0.4, -0.02));
}

#[test]
fn should_recover_a_group_far_smaller_than_half_precision_resolves_well() {
    // The scale of this group is a subnormal half, the coarsest scale there
    // is: rounding it down would clip the far end of the group.
    assert_recovered_within_half_a_step(&evenly_spread_group(-0.001, 0.001));
    assert_recovered_within_half_a_step(&evenly_spread_group(0.000_01, 0.000_02));
}

#[test]
fn should_recover_the_extremes_of_every_group_without_clipping() {
    for (lowest_weight, highest_weight) in [(-0.731, 0.412), (-0.0517, 0.0933), (0.25, 3.5)] {
        let group_weights = evenly_spread_group(lowest_weight, highest_weight);
        let quantized_matrix = quantize_matrix(&group_weights).expect("the group should quantize");
        let step = half_bits_to_f32(quantized_matrix.scale_half_bits[0]).abs();
        let recovered_weights = dequantized_weights(&group_weights);

        assert!((recovered_weights[0] - lowest_weight).abs() <= step * 0.5 + f32::EPSILON);
        assert!(
            (recovered_weights[QUANTIZATION_GROUP_SIZE - 1] - highest_weight).abs()
                <= step * 0.5 + f32::EPSILON
        );
    }
}

#[test]
fn should_quantize_a_group_of_equal_weights_without_dividing_by_zero() {
    let recovered_weights = dequantized_weights(&[0.125; QUANTIZATION_GROUP_SIZE]);

    for recovered_weight in recovered_weights {
        assert!((recovered_weight - 0.125).abs() < 1.0e-3);
    }
}

#[test]
fn should_quantize_an_all_zero_group_to_exact_zeros() {
    let recovered_weights = dequantized_weights(&[0.0; QUANTIZATION_GROUP_SIZE]);

    assert!(recovered_weights.iter().all(|weight| *weight == 0.0));
}

#[test]
fn should_give_each_group_its_own_scale_and_bias() {
    let mut two_groups = evenly_spread_group(-1.0, 1.0);
    two_groups.extend(evenly_spread_group(-0.001, 0.001));

    let quantized_matrix = quantize_matrix(&two_groups).expect("two groups should quantize");

    assert_eq!(quantized_matrix.scale_half_bits.len(), 2);
    assert_eq!(quantized_matrix.quantized_levels.len(), 64);
    assert_ne!(
        quantized_matrix.scale_half_bits[0],
        quantized_matrix.scale_half_bits[1]
    );
    assert_recovered_within_half_a_step(&two_groups);
}

#[test]
fn should_refuse_weights_that_do_not_fill_whole_groups() {
    assert!(quantize_matrix(&[0.5; QUANTIZATION_GROUP_SIZE - 1]).is_err());
    assert!(quantize_matrix(&[]).is_err());
}

#[test]
fn should_refuse_a_weight_that_is_not_finite() {
    let mut group_weights = evenly_spread_group(-0.5, 0.5);
    group_weights[3] = f32::NAN;

    assert!(quantize_matrix(&group_weights).is_err());
}
