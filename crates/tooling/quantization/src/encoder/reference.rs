//! Pure float32 reference encoder: the oracle for the GPU implementation.
//!
//! Works on one row-major 2-D slab `[rows, columns]`; a 3-D tensor is
//! encoded as `n_experts` slabs. All arithmetic is float32 so the reference
//! and the Metal implementation can be compared bit for bit on the codes and
//! within float rounding on scales and biases.

use crate::bit_width::{BitWidth, GroupSize};
use crate::encoder::range_search_policy::{RANGE_SEARCH_SCALE_FACTORS, SCALE_EPSILON};
use crate::error::QuantizationError;

#[cfg(test)]
#[path = "reference_tests.rs"]
mod reference_tests;

/// The encoder output for one slab, in MLX affine layout.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineQuantizedSlab {
    pub rows: usize,
    pub columns: usize,
    pub bit_width: BitWidth,
    pub group_size: GroupSize,
    /// `rows * columns * bits / 32` words, codes packed low-to-high.
    pub packed_words: Vec<u32>,
    /// `rows * columns / group_size` scales.
    pub scales: Vec<f32>,
    /// `rows * columns / group_size` biases.
    pub biases: Vec<f32>,
}

/// One group's chosen affine parameters and the error they achieve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupAffineParameters {
    pub scale: f32,
    pub bias: f32,
    pub squared_error: f32,
}

/// The search's starting point for one group, OptiQ's initialization in
/// `_search_affine`: `scale = (max - min) / (2^bits - 1)` floored at
/// [`super::SCALE_EPSILON`], `bias = min`. This is deliberately not what
/// `mx.quantize` does (see [`mlx_affine_parameters`]); the search seeds from
/// the plain range and only ever moves to a candidate that lowers the error.
#[must_use]
pub fn stock_min_max_parameters(group: &[f32], bit_width: BitWidth) -> GroupAffineParameters {
    let (group_minimum, group_maximum) = group_extremes(group);
    let scale = ((group_maximum - group_minimum) / levels_of(bit_width)).max(SCALE_EPSILON);
    let bias = group_minimum;
    GroupAffineParameters {
        scale,
        bias,
        squared_error: squared_error_with(group, scale, bias, bit_width, f32::round_ties_even),
    }
}

/// The parameters `mx.quantize(mode="affine")` itself chooses, which the
/// encoder must reproduce word for word when the search is off.
///
/// MLX anchors the range at whichever extreme has the larger magnitude and
/// snaps the scale so that edge lands exactly on a code; when the edge would
/// round to code zero the bias becomes exactly zero, which keeps a true zero
/// representable. The Metal kernel rounds half away from zero, unlike the
/// `mx.round` operation the search uses, so the codes are computed with
/// [`f32::round`] here.
#[must_use]
pub fn mlx_affine_parameters(group: &[f32], bit_width: BitWidth) -> GroupAffineParameters {
    let (group_minimum, group_maximum) = group_extremes(group);
    let unsigned_scale =
        ((group_maximum - group_minimum) / levels_of(bit_width)).max(SCALE_EPSILON);
    let is_minimum_the_wider_edge = group_minimum.abs() > group_maximum.abs();
    let (edge, signed_scale) = if is_minimum_the_wider_edge {
        (group_minimum, unsigned_scale)
    } else {
        (group_maximum, -unsigned_scale)
    };
    let edge_code = (edge / signed_scale).round();
    let is_edge_at_code_zero = edge_code == 0.0;
    let scale = if is_edge_at_code_zero {
        signed_scale
    } else {
        edge / edge_code
    };
    let bias = if is_edge_at_code_zero { 0.0 } else { edge };
    GroupAffineParameters {
        scale,
        bias,
        squared_error: squared_error_with(group, scale, bias, bit_width, f32::round),
    }
}

/// Squared reconstruction error of a group under the given parameters, with
/// codes rounded half to even (as `mx.round` does) and clipped to
/// `[0, 2^bits - 1]`.
#[must_use]
pub fn squared_error(group: &[f32], scale: f32, bias: f32, bit_width: BitWidth) -> f32 {
    squared_error_with(group, scale, bias, bit_width, f32::round_ties_even)
}

/// The range search for one group, OptiQ's `_search_affine`.
///
/// Starting from [`stock_min_max_parameters`], each extreme in turn anchors
/// a base scale snapped so that extreme lands on a code (with the bias set to
/// exactly zero when it lands on code zero), and each factor in
/// [`super::RANGE_SEARCH_SCALE_FACTORS`] scales that base. A candidate is
/// taken only when its squared error is strictly lower than the best so far,
/// so the search never does worse than its starting point.
#[must_use]
pub fn search_group_parameters(group: &[f32], bit_width: BitWidth) -> GroupAffineParameters {
    let (group_minimum, group_maximum) = group_extremes(group);
    let mut best_parameters = stock_min_max_parameters(group, bit_width);
    let anchors = [
        (group_maximum, group_minimum, -1.0_f32),
        (group_minimum, group_maximum, 1.0_f32),
    ];
    for (edge, opposite_edge, direction_sign) in anchors {
        let (base_scale, anchored_bias) =
            anchored_scale_and_bias(edge, opposite_edge, direction_sign, bit_width);
        for scale_factor in RANGE_SEARCH_SCALE_FACTORS {
            let candidate_scale = base_scale * scale_factor;
            let candidate_error = squared_error(group, candidate_scale, anchored_bias, bit_width);
            if candidate_error < best_parameters.squared_error {
                best_parameters = GroupAffineParameters {
                    scale: candidate_scale,
                    bias: anchored_bias,
                    squared_error: candidate_error,
                };
            }
        }
    }
    best_parameters
}

/// Packs one slab's codes exactly as `mx.quantize(mode="affine")` lays them
/// out: a little-endian bit stream, each code occupying `bits` consecutive
/// bits starting at bit `index * bits`. For 2, 4, and 8 bits that is
/// `32 / bits` whole codes per word; for 3, 5, and 6 bits a code may
/// straddle two words, which is what OptiQ's bit-expansion path produces.
/// Rows are contiguous because `columns * bits` is a multiple of 32.
#[must_use]
pub fn pack_affine_codes(codes: &[u32], columns: usize, bit_width: BitWidth) -> Vec<u32> {
    let code_bits = bit_width.bits() as usize;
    let rows = codes.len() / columns.max(1);
    let mut packed_words = vec![0_u32; rows * columns * code_bits / 32];
    for (code_index, code) in codes.iter().enumerate() {
        let first_bit = code_index * code_bits;
        for bit_offset in 0..code_bits {
            if (code >> bit_offset) & 1 == 1 {
                let stream_bit = first_bit + bit_offset;
                packed_words[stream_bit / 32] |= 1 << (stream_bit % 32);
            }
        }
    }
    packed_words
}

/// Unpacks codes written by [`pack_affine_codes`], for round-trip tests and
/// for [`dequantize_slab`].
#[must_use]
pub fn unpack_affine_codes(
    packed_words: &[u32],
    rows: usize,
    columns: usize,
    bit_width: BitWidth,
) -> Vec<u32> {
    let code_bits = bit_width.bits() as usize;
    let mut codes = vec![0_u32; rows * columns];
    for (code_index, code) in codes.iter_mut().enumerate() {
        let first_bit = code_index * code_bits;
        for bit_offset in 0..code_bits {
            let stream_bit = first_bit + bit_offset;
            let bit_value = (packed_words[stream_bit / 32] >> (stream_bit % 32)) & 1;
            *code |= bit_value << bit_offset;
        }
    }
    codes
}

/// Encodes a slab with the range search when `use_range_search` is set and
/// with `mx.quantize`'s own parameters otherwise. Whether the search applies
/// to a width is the caller's decision through
/// [`super::RangeSearchPolicy::applies_to`].
pub fn encode_slab(
    weights: &[f32],
    rows: usize,
    columns: usize,
    bit_width: BitWidth,
    group_size: GroupSize,
    use_range_search: bool,
) -> Result<AffineQuantizedSlab, QuantizationError> {
    validate_slab_shape(weights, rows, columns, bit_width, group_size)?;
    let group_elements = group_size.elements() as usize;
    let group_count = weights.len() / group_elements;
    let mut scales = Vec::with_capacity(group_count);
    let mut biases = Vec::with_capacity(group_count);
    let mut codes = Vec::with_capacity(weights.len());
    for group in weights.chunks_exact(group_elements) {
        let (parameters, round_code) = if use_range_search {
            let rounding: fn(f32) -> f32 = f32::round_ties_even;
            (search_group_parameters(group, bit_width), rounding)
        } else {
            let rounding: fn(f32) -> f32 = f32::round;
            (mlx_affine_parameters(group, bit_width), rounding)
        };
        append_group_codes(&mut codes, group, parameters, bit_width, round_code);
        scales.push(parameters.scale);
        biases.push(parameters.bias);
    }
    Ok(AffineQuantizedSlab {
        rows,
        columns,
        bit_width,
        group_size,
        packed_words: pack_affine_codes(&codes, columns, bit_width),
        scales,
        biases,
    })
}

/// Dequantizes a slab back to float32 for error measurement:
/// `code * scale + bias` per group.
#[must_use]
pub fn dequantize_slab(slab: &AffineQuantizedSlab) -> Vec<f32> {
    let codes = unpack_affine_codes(&slab.packed_words, slab.rows, slab.columns, slab.bit_width);
    let group_elements = slab.group_size.elements() as usize;
    let mut reconstructed_weights = Vec::with_capacity(codes.len());
    for (group_index, group_codes) in codes.chunks_exact(group_elements).enumerate() {
        let scale = slab.scales[group_index];
        let bias = slab.biases[group_index];
        for code in group_codes {
            reconstructed_weights.push(*code as f32 * scale + bias);
        }
    }
    reconstructed_weights
}

/// Fraction of exactly-zero values, the distribution statistic OptiQ used to
/// detect the 2-bit collapse (0.50 against a 0.13 source). Zero for an empty
/// slice.
#[must_use]
pub fn exact_zero_fraction(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut zero_count = 0_usize;
    for value in values {
        if *value == 0.0 {
            zero_count += 1;
        }
    }
    zero_count as f64 / values.len() as f64
}

/// `2^bits - 1`, the highest code, as a float.
fn levels_of(bit_width: BitWidth) -> f32 {
    ((1_u32 << bit_width.bits()) - 1) as f32
}

/// The group's minimum and maximum.
fn group_extremes(group: &[f32]) -> (f32, f32) {
    let mut group_minimum = f32::INFINITY;
    let mut group_maximum = f32::NEG_INFINITY;
    for weight in group {
        group_minimum = group_minimum.min(*weight);
        group_maximum = group_maximum.max(*weight);
    }
    (group_minimum, group_maximum)
}

/// One weight's clipped code under the given parameters and rounding rule.
fn code_of(weight: f32, scale: f32, bias: f32, levels: f32, round_code: fn(f32) -> f32) -> f32 {
    round_code((weight - bias) / scale).clamp(0.0, levels)
}

/// The squared error sum with an explicit rounding rule, shared by the
/// search (half to even) and the `mx.quantize` path (half away from zero).
fn squared_error_with(
    group: &[f32],
    scale: f32,
    bias: f32,
    bit_width: BitWidth,
    round_code: fn(f32) -> f32,
) -> f32 {
    let levels = levels_of(bit_width);
    let mut error_sum = 0.0_f32;
    for weight in group {
        let reconstructed = code_of(*weight, scale, bias, levels, round_code) * scale + bias;
        error_sum += (weight - reconstructed) * (weight - reconstructed);
    }
    error_sum
}

/// The base scale and bias for one search anchor: the edge snapped onto a
/// code, with the bias forced to exactly zero when that code is zero.
fn anchored_scale_and_bias(
    edge: f32,
    opposite_edge: f32,
    direction_sign: f32,
    bit_width: BitWidth,
) -> (f32, f32) {
    let raw_scale =
        ((edge - opposite_edge).abs() / levels_of(bit_width)).max(SCALE_EPSILON) * direction_sign;
    let edge_code = (edge / raw_scale).round_ties_even();
    if edge_code == 0.0 {
        (raw_scale, 0.0)
    } else {
        (edge / edge_code, edge)
    }
}

/// Appends one group's clipped codes.
fn append_group_codes(
    codes: &mut Vec<u32>,
    group: &[f32],
    parameters: GroupAffineParameters,
    bit_width: BitWidth,
    round_code: fn(f32) -> f32,
) {
    let levels = levels_of(bit_width);
    for weight in group {
        let code = code_of(
            *weight,
            parameters.scale,
            parameters.bias,
            levels,
            round_code,
        );
        codes.push(code as u32);
    }
}

/// Refuses slabs the affine layout cannot hold: a weight count that does not
/// match the shape, a row that does not divide into groups or into whole
/// words, or the keep-bfloat16 tier, which is never encoded.
fn validate_slab_shape(
    weights: &[f32],
    rows: usize,
    columns: usize,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<(), QuantizationError> {
    if bit_width.keeps_bfloat16() {
        return Err(QuantizationError::UnsupportedBitWidth {
            bits: bit_width.bits(),
        });
    }
    let group_elements = group_size.elements() as usize;
    let is_shape_consistent = weights.len() == rows * columns;
    let divides_into_groups = columns.is_multiple_of(group_elements);
    let divides_into_words = (columns * bit_width.bits() as usize) % 32 == 0;
    if is_shape_consistent && divides_into_groups && divides_into_words {
        return Ok(());
    }
    Err(QuantizationError::TensorNotGroupQuantizable {
        tensor_name: "slab".to_owned(),
        shape: vec![rows, columns],
        group_size: group_size.elements(),
    })
}
