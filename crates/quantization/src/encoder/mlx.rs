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

use astronomical_runtime_integration::MlxRuntime;
use mlx_c_rust::MlxArray;

use crate::bit_width::{BitWidth, GroupSize};
use crate::error::QuantizationError;

use super::RangeSearchPolicy;

/// The three MLX arrays of an affine-quantized tensor.
#[derive(Debug)]
pub struct MlxAffineQuantizedTensor {
    pub packed_weight: MlxArray,
    pub scales: MlxArray,
    pub biases: MlxArray,
    pub bit_width: BitWidth,
    pub group_size: GroupSize,
}

/// Encodes `weights` (bf16, f16, or f32; 2-D or 3-D) at `bit_width`.
///
/// Work to port (OptiQ's `quantize_best`):
/// 1. `bit_width.bits() > 4` or policy does not apply → `runtime.quantize_affine`.
/// 2. 3-D and larger than the chunk limit → slice along axis 0 into pieces
///    of `max(1, shape[0] * chunk / size)`, encode each with
///    [`search_affine_whole`], concatenate along axis 0, evaluate, and drain
///    the allocator cache.
/// 3. Otherwise [`search_affine_whole`].
pub fn quantize_best(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
    policy: RangeSearchPolicy,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    let _ = (runtime, weights, bit_width, group_size, policy);
    todo!("dispatch between stock quantize_affine, chunked search, and whole-tensor search")
}

/// The per-group range search over one tensor small enough to hold its
/// float32 working copy.
///
/// Work to port (OptiQ's `_search_affine`): reshape to
/// `[-1, columns / group_size, group_size]` as float32; compute per-group
/// min and max with keep-dims; initialize best scale, bias, and error from
/// stock min/max; for each anchor and factor compute the candidate scale,
/// clip-rounded codes, squared error, and `where(error < best, candidate,
/// best)` for all three, evaluating after every candidate; then pack with
/// [`pack_affine_codes_mlx`], reshape scales and biases to
/// `[..., columns / group_size]`, cast them to the input dtype, evaluate,
/// and drain the allocator cache.
pub fn search_affine_whole(
    runtime: &MlxRuntime,
    weights: &MlxArray,
    bit_width: BitWidth,
    group_size: GroupSize,
) -> Result<MlxAffineQuantizedTensor, QuantizationError> {
    let _ = (runtime, weights, bit_width, group_size);
    todo!("run the anchored, seven-factor search as array operations")
}

/// Packs float32 codes into `uint32` words exactly as `mx.quantize` does.
///
/// Work to port (OptiQ's `_pack_affine_codes`): for 2, 4, 8 bits multiply
/// each run of `32 / bits` codes by `2^(bits * position)` and sum along the
/// run; for 3, 5, 6 bits expand each code to its bits with `right_shift` and
/// `bitwise_and`, reshape to runs of 32, and sum `left_shift(bit, position)`.
/// Output shape is `[..., columns * bits / 32]`.
pub fn pack_affine_codes_mlx(
    runtime: &MlxRuntime,
    codes: &MlxArray,
    original_shape: &[i32],
    bit_width: BitWidth,
) -> Result<MlxArray, QuantizationError> {
    let _ = (runtime, codes, original_shape, bit_width);
    todo!("pack with the power-of-two or bit-expansion path")
}
