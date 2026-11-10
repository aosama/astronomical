//! A tensor's bytes as numbers and back: the little-endian encodings a
//! safetensors file stores, read into `f32` for arithmetic and written out
//! in half precision.

use super::half_float::{f32_to_half_bits, half_bits_to_f32};
use super::safetensors_file::TensorElementType;

/// The tensor's values as `f32`.
pub(crate) fn decoded_weights(
    source_bytes: &[u8],
    element_type: TensorElementType,
) -> Result<Vec<f32>, String> {
    match element_type {
        TensorElementType::Float16 => {
            let mut weights = Vec::with_capacity(source_bytes.len() / 2);
            let (whole_halves, _) = source_bytes.as_chunks::<2>();
            for half_bytes in whole_halves {
                weights.push(half_bits_to_f32(u16::from_le_bytes(*half_bytes)));
            }
            Ok(weights)
        }
        TensorElementType::Float32 => {
            let mut weights = Vec::with_capacity(source_bytes.len() / 4);
            let (whole_singles, _) = source_bytes.as_chunks::<4>();
            for single_bytes in whole_singles {
                weights.push(f32::from_le_bytes(*single_bytes));
            }
            Ok(weights)
        }
        TensorElementType::Unsigned32 => {
            Err("a PyTorch Laya checkpoint holds no integer tensors".to_owned())
        }
    }
}

/// The tensor's bytes in half precision: unchanged when it already is.
pub(crate) fn half_precision_bytes(
    source_bytes: Vec<u8>,
    element_type: TensorElementType,
) -> Result<Vec<u8>, String> {
    if element_type == TensorElementType::Float16 {
        return Ok(source_bytes);
    }
    let weights = decoded_weights(&source_bytes, element_type)?;
    let mut half_bits = Vec::with_capacity(weights.len());
    for weight in weights {
        half_bits.push(f32_to_half_bits(weight));
    }
    Ok(half_bits_bytes(&half_bits))
}

/// Half-precision bits as the little-endian bytes a safetensors file holds.
pub(crate) fn half_bits_bytes(half_bits: &[u16]) -> Vec<u8> {
    let mut half_bytes = Vec::with_capacity(half_bits.len() * 2);
    for bits in half_bits {
        half_bytes.extend_from_slice(&bits.to_le_bytes());
    }
    half_bytes
}

#[cfg(test)]
#[path = "tensor_bytes_tests.rs"]
mod tests;
