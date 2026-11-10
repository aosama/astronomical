//! Eight-bit affine quantization of one weight matrix, in the layout MLX
//! reads.
//!
//! Each row is cut into groups of 32 weights. A group keeps one scale and
//! one bias, and every weight becomes one byte `q` with
//! `weight ≈ q * scale + bias`. Four bytes are packed into each `u32`, lowest
//! byte first, which is exactly the little-endian byte order of the `u32`,
//! so the packed tensor's bytes are simply the `q` values in order.
//!
//! The scale and bias are chosen as MLX's own `quantize` chooses them. They
//! are then rounded to half precision, because that is how they are stored,
//! and the weights are quantized against the stored values. Rounding a scale
//! down would leave the 256 levels short of the group's far end and clip the
//! weights there, so a scale that falls short is widened by one
//! half-precision step. Every weight is therefore recovered to within half
//! a step of the scale the engine actually multiplies by.

use super::half_float::{f32_to_half_bits, half_bits_to_f32};

/// Weights that share one scale and one bias.
pub(crate) const QUANTIZATION_GROUP_SIZE: usize = 32;
pub(crate) const QUANTIZATION_BITS: usize = 8;
/// The largest value one quantized weight can take.
const HIGHEST_QUANTIZED_LEVEL: f32 = 255.0;
/// MLX's floor for a scale, so a group of equal weights does not divide by zero.
const MINIMUM_SCALE_MAGNITUDE: f32 = 1.0e-7;

/// One quantized matrix: a byte per weight, and a scale and bias per group.
#[derive(Debug, PartialEq)]
pub(crate) struct QuantizedMatrix {
    /// One byte per source weight, row by row.
    pub(crate) quantized_levels: Vec<u8>,
    /// Half-precision bits of each group's scale, row by row.
    pub(crate) scale_half_bits: Vec<u16>,
    /// Half-precision bits of each group's bias, row by row.
    pub(crate) bias_half_bits: Vec<u16>,
}

/// Quantizes `weights`, whose length must be a whole number of groups.
pub(crate) fn quantize_matrix(weights: &[f32]) -> Result<QuantizedMatrix, String> {
    if weights.is_empty() || !weights.len().is_multiple_of(QUANTIZATION_GROUP_SIZE) {
        return Err(format!(
            "a quantized matrix needs a whole number of {QUANTIZATION_GROUP_SIZE}-weight groups, \
             got {} weights",
            weights.len()
        ));
    }
    let group_count = weights.len() / QUANTIZATION_GROUP_SIZE;
    let mut quantized_matrix = QuantizedMatrix {
        quantized_levels: Vec::with_capacity(weights.len()),
        scale_half_bits: Vec::with_capacity(group_count),
        bias_half_bits: Vec::with_capacity(group_count),
    };
    let (whole_groups, _) = weights.as_chunks::<QUANTIZATION_GROUP_SIZE>();
    for group_weights in whole_groups {
        let (scale_bits, bias_bits) = group_scale_and_bias_half_bits(group_weights)?;
        let stored_scale = half_bits_to_f32(scale_bits);
        let stored_bias = half_bits_to_f32(bias_bits);
        for weight in group_weights {
            let quantized_level = ((weight - stored_bias) / stored_scale)
                .round_ties_even()
                .clamp(0.0, HIGHEST_QUANTIZED_LEVEL);
            quantized_matrix
                .quantized_levels
                .push(quantized_level as u8);
        }
        quantized_matrix.scale_half_bits.push(scale_bits);
        quantized_matrix.bias_half_bits.push(bias_bits);
    }
    Ok(quantized_matrix)
}

/// The scale and bias of one group, as the half-precision bits stored.
///
/// The bias is the group's extreme of larger magnitude, and the scale is
/// signed so that the quantized levels run from that extreme towards the
/// other one. Adjusting the scale to `edge / round(edge / scale)` puts zero
/// on a level whenever the group spans zero, up to the rounding of the
/// stored scale.
fn group_scale_and_bias_half_bits(group_weights: &[f32]) -> Result<(u16, u16), String> {
    let mut lowest_weight = f32::INFINITY;
    let mut highest_weight = f32::NEG_INFINITY;
    for weight in group_weights {
        if !weight.is_finite() {
            return Err("the checkpoint holds a weight that is not a finite number".to_owned());
        }
        lowest_weight = lowest_weight.min(*weight);
        highest_weight = highest_weight.max(*weight);
    }
    let lowest_is_the_larger_extreme = lowest_weight.abs() > highest_weight.abs();
    let step_magnitude =
        ((highest_weight - lowest_weight) / HIGHEST_QUANTIZED_LEVEL).max(MINIMUM_SCALE_MAGNITUDE);
    let (edge, mut scale) = if lowest_is_the_larger_extreme {
        (lowest_weight, step_magnitude)
    } else {
        (highest_weight, -step_magnitude)
    };
    let edge_level = (edge / scale).round_ties_even();
    let bias = if edge_level == 0.0 {
        0.0
    } else {
        scale = edge / edge_level;
        edge
    };
    let bias_bits = f32_to_half_bits(bias);
    let stored_bias = half_bits_to_f32(bias_bits);
    let farthest_distance_from_bias = (lowest_weight - stored_bias)
        .abs()
        .max((highest_weight - stored_bias).abs());
    Ok((
        covering_scale_half_bits(scale, farthest_distance_from_bias),
        bias_bits,
    ))
}

/// The half-precision scale nearest to `scale` whose 256 levels still reach
/// `farthest_distance_from_bias`; one half-precision step wider than the
/// nearest when the nearest falls short.
fn covering_scale_half_bits(scale: f32, farthest_distance_from_bias: f32) -> u16 {
    let nearest_scale_bits = f32_to_half_bits(scale);
    let reach = half_bits_to_f32(nearest_scale_bits).abs() * HIGHEST_QUANTIZED_LEVEL;
    if reach >= farthest_distance_from_bias {
        return nearest_scale_bits;
    }
    // Adding one to the bits of a finite half moves its magnitude up by one
    // step, across exponent boundaries too, and leaves the sign bit alone.
    nearest_scale_bits + 1
}

#[cfg(test)]
#[path = "affine_quantization_tests.rs"]
mod tests;
