//! The MLX affine 4-bit scheme every packed projection in this crate uses:
//! eight 4-bit values per uint32 word, one scale and one offset per 64 inputs.
//! Kept free of MLX so artifact validation can check packed shapes without the
//! runtime.

/// Input channels that share one scale and one offset.
pub(crate) const AFFINE_QUANTIZATION_GROUP_SIZE: usize = 64;
/// Bits per quantized value.
pub(crate) const AFFINE_QUANTIZATION_BITS: i32 = 4;
/// Four-bit values packed into each uint32 word: the divisor every packed
/// weight's inner axis carries (`in / 8`).
pub(crate) const AFFINE_QUANTIZED_VALUES_PER_WORD: usize = 8;
