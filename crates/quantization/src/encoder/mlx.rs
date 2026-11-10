//! The MLX (Metal) encoder: the same search as [`super::reference`], run as
//! whole-tensor array operations on the GPU.
//!
//! Design notes for the implementation:
//!
//! - Every candidate is evaluated eagerly (`evaluate_arrays` after each
//!   factor). Deferring to the end of the loop leaves seven tensor-sized
//!   graphs live at once and the kernel kills the process on a large expert
//!   tensor; this is OptiQ's hard-won rule, not an optimization choice.
//! - Widths above 4 bypass the search and call `quantize_affine` directly.
//! - A 3-D tensor larger than [`super::RANGE_SEARCH_CHUNK_ELEMENTS`] is
//!   encoded in slices along its leading axis and concatenated.
//! - The float32 working copy of the grouped tensor is the peak allocation:
//!   `4 * elements` bytes plus candidate error/scale/bias arrays of
//!   `elements / group_size`. The requantizer sizes its tensor budget from
//!   this figure.
//! - Where the Rust port can improve on the Python: the search over seven
//!   factors is embarrassingly parallel per group and could be one fused
//!   Metal kernel instead of seven array passes; measure before fusing, and
//!   keep the reference encoder as the correctness oracle for any kernel.
//!
//! Ported from OptiQ's `quantize_best`, `_search_affine`, and `_pack_affine_codes` (`optiq/core/lowbit.py`).

use astronomical_runtime_integration::MlxRuntime;
use mlx_bridge::{MlxArray, MlxDtype};

use crate::bit_width::{BitWidth, GroupSize};
use crate::error::QuantizationError;

use super::{
    RANGE_SEARCH_CHUNK_ELEMENTS, RANGE_SEARCH_SCALE_FACTORS, RangeSearchPolicy, SCALE_EPSILON,
};

/// The three MLX arrays of an affine-quantized tensor.
#[derive(Debug)]
pub struct MlxAffineQuantizedTensor {
    pub packed_weight: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
    pub bit_width: BitWidth,
    pub group_size: GroupSize,
}

/// Encodes `weights` (bf16, f16, or f32; 2-D or 3-D) at `bit_width`,
/// OptiQ's `quantize_best`.
///
/// Widths the policy does not search go straight to stock `quantize_affine`
/// so those artifacts stay byte-identical to `mx.quantize`. A 3-D tensor
/// larger than [`RANGE_SEARCH_CHUNK_ELEMENTS`] is searched in slices along
/// its leading axis; groups run along the last axis, so slicing changes
/// nothing about the result and only bounds the float32 working copy.
pub fn quantize_best(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
    policy: RangeSearchPolicy,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    validate_encodable(weights, bit_width, group_size)?;
    if !policy.applies_to(bit_width) {
        let (packed_weight, scales, biases) = runtime.quantize_affine(
            weights,
            group_size.elements() as i32,
            bit_width.bits() as i32,
        )?;
        return Ok(MlxAffineQuantizedTensor {
            packed_weight,
            scales,
            biases,
            bit_width,
            group_size,
        });
    }
    let original_shape = weights.shape();
    if original_shape.len() >= 3 && weights.element_count() > RANGE_SEARCH_CHUNK_ELEMENTS {
        return search_affine_in_leading_axis_slices(runtime, weights, bit_width, group_size);
    }
    search_affine_whole(runtime, weights, bit_width, group_size)
}

/// The per-group range search over one tensor small enough to hold its
/// float32 working copy, OptiQ's `_search_affine`.
///
/// Every candidate is evaluated before the next is built; see the module
/// notes for why deferring is fatal on large expert tensors.
pub fn search_affine_whole(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    validate_encodable(weights, bit_width, group_size)?;
    let original_shape = weights.shape();
    let group_elements = group_size.elements() as i32;
    let columns = original_shape[original_shape.len() - 1];
    let grouped_weights = runtime.astype(
        &runtime.reshape(weights, &[-1, columns / group_elements, group_elements])?,
        MlxDtype::Float32,
    )?;
    let levels = MlxArray::from_scalar_f32(levels_of(bit_width));
    let (group_extremes, mut best_candidate) =
        seed_from_plain_min_max(runtime, &grouped_weights, &levels)?;
    for anchor in search_anchors(&group_extremes) {
        let (base_scale, anchored_bias) = anchored_scale_and_bias(runtime, &anchor, &levels)?;
        for scale_factor in RANGE_SEARCH_SCALE_FACTORS {
            let candidate_scale =
                runtime.multiply(&base_scale, &MlxArray::from_scalar_f32(scale_factor))?;
            best_candidate = keep_better_candidate(
                runtime,
                &grouped_weights,
                &levels,
                best_candidate,
                candidate_scale,
                &anchored_bias,
            )?;
        }
    }
    finish_search(
        runtime,
        weights,
        &grouped_weights,
        best_candidate,
        bit_width,
        group_size,
    )
}

/// Packs float32 codes into `uint32` words exactly as `mx.quantize` does,
/// OptiQ's `_pack_affine_codes`.
///
/// For 2, 4, and 8 bits each run of `32 / bits` codes is multiplied by its
/// power of two and summed. For 3, 5, and 6 bits a code may straddle two
/// words, so each code is expanded to its bits, the bit stream regrouped in
/// runs of 32, and each bit shifted to its position; the sum is a bitwise or
/// because no two terms share a position. Output shape is
/// `[..., columns * bits / 32]`.
pub fn pack_affine_codes_mlx(
    runtime: &MlxRuntime,
    codes: &MlxArray,
    original_shape: &[i32],
    bit_width: BitWidth,
) -> Result<MlxArray, QuantizationError> {
    let code_bits = bit_width.bits();
    let integer_codes = runtime.astype(codes, MlxDtype::UInt32)?;
    let flattened_rows = integer_codes.shape()[0];
    let packed_words = if 32 % code_bits == 0 {
        let codes_per_word = (32 / code_bits) as i32;
        let word_runs = runtime.reshape(&integer_codes, &[flattened_rows, -1, codes_per_word])?;
        let positional_weights = power_of_two_multipliers(runtime, code_bits)?;
        runtime.sum_axis(
            &runtime.multiply(&word_runs, &positional_weights)?,
            2,
            false,
        )?
    } else {
        let code_bit_stream = expand_codes_to_bits(runtime, &integer_codes, code_bits)?;
        let word_runs = runtime.reshape(&code_bit_stream, &[flattened_rows, -1, 32])?;
        let bit_positions = runtime.array_from_u32(&(0..32).collect::<Vec<u32>>(), &[32])?;
        runtime.sum_axis(&runtime.left_shift(&word_runs, &bit_positions)?, -1, false)?
    };
    let mut packed_shape = original_shape.to_vec();
    let last_axis = packed_shape.len() - 1;
    packed_shape[last_axis] = original_shape[last_axis] * code_bits as i32 / 32;
    Ok(runtime.reshape(&packed_words, &packed_shape)?)
}

/// Every group's minimum and maximum, the edges the search anchors on.
struct GroupExtremes {
    group_minimum: MlxArray,
    group_maximum: MlxArray,
}

/// The best parameters and error so far, for every group at once.
struct GroupSearchState {
    best_scale: MlxArray,
    best_bias: MlxArray,
    best_squared_error: MlxArray,
}

/// One anchor: the edge snapped onto a code, the opposite edge, and the sign
/// that points the scale from the edge toward the opposite edge.
struct SearchAnchor<'extremes> {
    edge: &'extremes MlxArray,
    opposite_edge: &'extremes MlxArray,
    direction_sign: f32,
}

/// The search seeds from the plain range (`scale = (max - min) / levels`,
/// `bias = min`), not from what `mx.quantize` picks; see
/// [`super::reference::stock_min_max_parameters`].
fn seed_from_plain_min_max(
    runtime: &MlxRuntime,
    grouped_weights: &MlxArray,
    levels: &MlxArray,
) -> Result<(GroupExtremes, GroupSearchState), QuantizationError> {
    let group_minimum = runtime.min_axis(grouped_weights, -1, true)?;
    let group_maximum = runtime.max_axis(grouped_weights, -1, true)?;
    let plain_scale = runtime.maximum(
        &runtime.divide(&runtime.subtract(&group_maximum, &group_minimum)?, levels)?,
        &MlxArray::from_scalar_f32(SCALE_EPSILON),
    )?;
    let best_bias = runtime.copy(&group_minimum)?;
    let best_squared_error =
        squared_error_per_group(runtime, grouped_weights, &plain_scale, &best_bias, levels)?;
    runtime.evaluate_arrays(&[&best_squared_error, &plain_scale, &best_bias])?;
    Ok((
        GroupExtremes {
            group_minimum,
            group_maximum,
        },
        GroupSearchState {
            best_scale: plain_scale,
            best_bias,
            best_squared_error,
        },
    ))
}

/// Anchored at the maximum first, then the minimum, as OptiQ orders them;
/// the order matters because ties keep the earlier candidate.
fn search_anchors(state: &GroupExtremes) -> [SearchAnchor<'_>; 2] {
    [
        SearchAnchor {
            edge: &state.group_maximum,
            opposite_edge: &state.group_minimum,
            direction_sign: -1.0,
        },
        SearchAnchor {
            edge: &state.group_minimum,
            opposite_edge: &state.group_maximum,
            direction_sign: 1.0,
        },
    ]
}

/// The base scale with the edge landing exactly on a code, and the bias
/// forced to exactly zero when that code is zero so a true zero stays
/// representable.
fn anchored_scale_and_bias(
    runtime: &MlxRuntime,
    anchor: &SearchAnchor<'_>,
    levels: &MlxArray,
) -> Result<(MlxArray, MlxArray), QuantizationError> {
    let range_scale = runtime.maximum(
        &runtime.divide(
            &runtime.abs(&runtime.subtract(anchor.edge, anchor.opposite_edge)?)?,
            levels,
        )?,
        &MlxArray::from_scalar_f32(SCALE_EPSILON),
    )?;
    let raw_scale = runtime.multiply(
        &range_scale,
        &MlxArray::from_scalar_f32(anchor.direction_sign),
    )?;
    let edge_code = runtime.round(&runtime.divide(anchor.edge, &raw_scale)?, 0)?;
    let zero = MlxArray::from_scalar_f32(0.0);
    let is_edge_on_nonzero_code = runtime.not_equal(&edge_code, &zero)?;
    let base_scale = runtime.where_select(
        &is_edge_on_nonzero_code,
        &runtime.divide(anchor.edge, &edge_code)?,
        &raw_scale,
    )?;
    let anchored_bias = runtime.where_select(&is_edge_on_nonzero_code, anchor.edge, &zero)?;
    Ok((base_scale, anchored_bias))
}

/// Takes the candidate wherever it strictly lowers the group's error, then
/// evaluates so only one candidate graph is ever live.
fn keep_better_candidate(
    runtime: &MlxRuntime,
    grouped_weights: &MlxArray,
    levels: &MlxArray,
    best_candidate: GroupSearchState,
    candidate_scale: MlxArray,
    candidate_bias: &MlxArray,
) -> Result<GroupSearchState, QuantizationError> {
    let candidate_error = squared_error_per_group(
        runtime,
        grouped_weights,
        &candidate_scale,
        candidate_bias,
        levels,
    )?;
    let is_improvement = runtime.less(&candidate_error, &best_candidate.best_squared_error)?;
    let best_squared_error = runtime.where_select(
        &is_improvement,
        &candidate_error,
        &best_candidate.best_squared_error,
    )?;
    let best_scale = runtime.where_select(
        &is_improvement,
        &candidate_scale,
        &best_candidate.best_scale,
    )?;
    let best_bias =
        runtime.where_select(&is_improvement, candidate_bias, &best_candidate.best_bias)?;
    runtime.evaluate_arrays(&[&best_squared_error, &best_scale, &best_bias])?;
    Ok(GroupSearchState {
        best_scale,
        best_bias,
        best_squared_error,
    })
}

/// Recomputes the codes from the chosen parameters, packs them, shapes the
/// scales and biases as `[..., columns / group_size]` in the input dtype,
/// and drains the allocator cache the search filled.
fn finish_search(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    grouped_weights: &MlxArray,
    best_candidate: GroupSearchState,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    let original_shape = weights.shape();
    let levels = MlxArray::from_scalar_f32(levels_of(bit_width));
    let codes = clipped_codes(
        runtime,
        grouped_weights,
        &best_candidate.best_scale,
        &best_candidate.best_bias,
        &levels,
    )?;
    let packed_weight = pack_affine_codes_mlx(runtime, &codes, &original_shape, bit_width)?;
    let mut parameter_shape = original_shape.clone();
    let last_axis = parameter_shape.len() - 1;
    parameter_shape[last_axis] /= group_size.elements() as i32;
    let scales = runtime.astype(
        &runtime.reshape(&best_candidate.best_scale, &parameter_shape)?,
        weights.dtype(),
    )?;
    let biases = runtime.astype(
        &runtime.reshape(&best_candidate.best_bias, &parameter_shape)?,
        weights.dtype(),
    )?;
    runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
    drop(codes);
    runtime.clear_allocator_cache()?;
    Ok(MlxAffineQuantizedTensor {
        packed_weight,
        scales,
        biases,
        bit_width,
        group_size,
    })
}

/// Searches a large 3-D tensor slice by slice along its leading axis and
/// concatenates the pieces.
fn search_affine_in_leading_axis_slices(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    let original_shape = weights.shape();
    let leading_extent = original_shape[0] as usize;
    let slice_extent =
        (leading_extent * RANGE_SEARCH_CHUNK_ELEMENTS / weights.element_count()).max(1);
    let mut encoded_slices = Vec::new();
    let mut slice_start = 0;
    while slice_start < leading_extent {
        let slice_stop = (slice_start + slice_extent).min(leading_extent);
        let mut starts = vec![0_i32; original_shape.len()];
        let mut stops = original_shape.clone();
        starts[0] = slice_start as i32;
        stops[0] = slice_stop as i32;
        let strides = vec![1_i32; original_shape.len()];
        let weight_slice = runtime.slice(weights, &starts, &stops, &strides)?;
        encoded_slices.push(search_affine_whole(
            runtime,
            &weight_slice,
            bit_width,
            group_size,
        )?);
        runtime.clear_allocator_cache()?;
        slice_start = slice_stop;
    }
    concatenate_encoded_slices(runtime, &encoded_slices, bit_width, group_size)
}

/// Joins slice encodings along the leading axis and evaluates the result.
fn concatenate_encoded_slices(
    runtime: &MlxRuntime,
    encoded_slices: &[MlxAffineQuantizedTensor],
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    let mut packed_slices = Vec::with_capacity(encoded_slices.len());
    let mut scale_slices = Vec::with_capacity(encoded_slices.len());
    let mut bias_slices = Vec::with_capacity(encoded_slices.len());
    for encoded_slice in encoded_slices {
        packed_slices.push(&encoded_slice.packed_weight);
        scale_slices.push(&encoded_slice.scales);
        bias_slices.push(&encoded_slice.biases);
    }
    let packed_weight = runtime.concatenate_axis(&packed_slices, 0)?;
    let scales = runtime.concatenate_axis(&scale_slices, 0)?;
    let biases = runtime.concatenate_axis(&bias_slices, 0)?;
    runtime.evaluate_arrays(&[&packed_weight, &scales, &biases])?;
    runtime.clear_allocator_cache()?;
    Ok(MlxAffineQuantizedTensor {
        packed_weight,
        scales,
        biases,
        bit_width,
        group_size,
    })
}

/// `clip(round((w - bias) / scale), 0, levels)`, rounding half to even as
/// `mx.round` does.
fn clipped_codes(
    runtime: &MlxRuntime,
    grouped_weights: &MlxArray,
    scale: &MlxArray,
    bias: &MlxArray,
    levels: &MlxArray,
) -> Result<MlxArray, QuantizationError> {
    let unclipped_codes = runtime.round(
        &runtime.divide(&runtime.subtract(grouped_weights, bias)?, scale)?,
        0,
    )?;
    let zero = MlxArray::from_scalar_f32(0.0);
    Ok(runtime.minimum(&runtime.maximum(&unclipped_codes, &zero)?, levels)?)
}

/// Per-group sum of squared reconstruction error, keeping the group axis.
fn squared_error_per_group(
    runtime: &MlxRuntime,
    grouped_weights: &MlxArray,
    scale: &MlxArray,
    bias: &MlxArray,
    levels: &MlxArray,
) -> Result<MlxArray, QuantizationError> {
    let codes = clipped_codes(runtime, grouped_weights, scale, bias, levels)?;
    let reconstruction = runtime.add(&runtime.multiply(&codes, scale)?, bias)?;
    let residual = runtime.subtract(grouped_weights, &reconstruction)?;
    // An exact square, where the Python encoder writes `residual ** 2`. On
    // Metal that is the `pow` kernel, which is not exact and does not round
    // the same way in every MLX build (the Python wheel's Metal library and
    // one compiled locally disagree in the last bit). The search keeps a
    // candidate only when it is strictly better, so that last bit decides
    // every near-tie: measured on Qwen3.5 0.8B, 480 of 5.87 million groups
    // at 2 and 3 bits. A square is the same on every build and device, so
    // this encoder is reproducible with itself; against OptiQ it differs in
    // those near-tie groups only, each time by an equal-error choice.
    Ok(runtime.sum_axis(&runtime.square(&residual)?, -1, true)?)
}

/// `[1, 2^bits, 2^(2 bits), ...]` as `uint32`, one per code in a word.
fn power_of_two_multipliers(
    runtime: &MlxRuntime,
    code_bits: u32,
) -> Result<MlxArray, QuantizationError> {
    let codes_per_word = 32 / code_bits;
    let mut multipliers = Vec::with_capacity(codes_per_word as usize);
    for code_position in 0..codes_per_word {
        multipliers.push(1_u32 << (code_bits * code_position));
    }
    Ok(runtime.array_from_u32(&multipliers, &[codes_per_word as i32])?)
}

/// Each code expanded to its `code_bits` bits, low bit first, as a trailing
/// axis.
fn expand_codes_to_bits(
    runtime: &MlxRuntime,
    integer_codes: &MlxArray,
    code_bits: u32,
) -> Result<MlxArray, QuantizationError> {
    let bit_offsets =
        runtime.array_from_u32(&(0..code_bits).collect::<Vec<u32>>(), &[code_bits as i32])?;
    let shifted_codes =
        runtime.right_shift(&runtime.expand_dims(integer_codes, -1)?, &bit_offsets)?;
    Ok(runtime.bitwise_and(&shifted_codes, &MlxArray::from_u32(&[1], &[])?)?)
}

/// `2^bits - 1`, the highest code.
fn levels_of(bit_width: BitWidth) -> f32 {
    ((1_u32 << bit_width.bits()) - 1) as f32
}

/// Refuses what the affine layout cannot hold: the keep-bfloat16 tier, a
/// rank other than 2 or 3, or a last axis that does not divide into groups
/// and whole words.
fn validate_encodable(
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<(), QuantizationError> {
    if bit_width.keeps_bfloat16() {
        return Err(QuantizationError::UnsupportedBitWidth {
            bits: bit_width.bits(),
        });
    }
    let shape = weights.shape();
    let columns = shape.last().copied().unwrap_or(0).max(0) as u32;
    let is_supported_rank = shape.len() == 2 || shape.len() == 3;
    let divides_into_groups = columns > 0 && columns.is_multiple_of(group_size.elements());
    let divides_into_words = (columns * bit_width.bits()).is_multiple_of(32);
    if is_supported_rank && divides_into_groups && divides_into_words {
        return Ok(());
    }
    let mut unsigned_shape = Vec::with_capacity(shape.len());
    for extent in shape {
        unsigned_shape.push(extent.max(0) as usize);
    }
    Err(QuantizationError::TensorNotGroupQuantizable {
        tensor_name: "encoder input".to_owned(),
        shape: unsigned_shape,
        group_size: group_size.elements(),
    })
}
