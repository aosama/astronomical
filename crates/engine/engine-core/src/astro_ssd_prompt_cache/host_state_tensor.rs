//! Decoder state copied out of MLX into memory the host owns, so an
//! `astro_ssd_prompt_cache` block can be written by a thread that never
//! touches MLX.
//!
//! An `MlxArray` must not cross to the background writer: evaluating a lazy
//! view, or freeing the last reference into the `mlx_buffer_cache`, from a
//! second thread races MLX's per-thread stream state. The engine thread
//! therefore evaluates a block's tensors once, copies each into a host
//! vector, and drops the arrays; the writer only ever sees these copies.

use std::collections::HashMap;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;

/// The elements of one host tensor, kept at their stored width so the copy
/// out of MLX is one pass and the writer converts to bytes as it writes.
#[derive(Clone, Debug, PartialEq)]
pub enum AstroSsdPromptCacheHostTensorElements {
    /// Float16 or bfloat16 bit patterns.
    SixteenBit(Vec<u16>),
    Float32(Vec<f32>),
    Int32(Vec<i32>),
}

/// One named decoder-state tensor held by the host.
#[derive(Clone, Debug, PartialEq)]
pub struct AstroSsdPromptCacheHostTensor {
    /// The safetensors dtype name the file header records (`F16`, `BF16`,
    /// `F32`, `I32`).
    pub safetensors_dtype_name: &'static str,
    pub shape: Vec<usize>,
    pub elements: AstroSsdPromptCacheHostTensorElements,
}

/// A block's tensors by their persistent names.
pub type AstroSsdPromptCacheHostTensors = HashMap<String, AstroSsdPromptCacheHostTensor>;

impl AstroSsdPromptCacheHostTensor {
    /// Bytes this tensor occupies in a safetensors payload.
    #[must_use]
    pub fn payload_byte_count(&self) -> usize {
        match &self.elements {
            AstroSsdPromptCacheHostTensorElements::SixteenBit(elements) => elements.len() * 2,
            AstroSsdPromptCacheHostTensorElements::Float32(elements) => elements.len() * 4,
            AstroSsdPromptCacheHostTensorElements::Int32(elements) => elements.len() * 4,
        }
    }

    /// Appends this tensor's little-endian payload bytes for elements
    /// `element_range` to `payload_bytes`.
    pub fn append_little_endian_bytes(
        &self,
        element_range: std::ops::Range<usize>,
        payload_bytes: &mut Vec<u8>,
    ) {
        match &self.elements {
            AstroSsdPromptCacheHostTensorElements::SixteenBit(elements) => {
                for element in &elements[element_range] {
                    payload_bytes.extend_from_slice(&element.to_le_bytes());
                }
            }
            AstroSsdPromptCacheHostTensorElements::Float32(elements) => {
                for element in &elements[element_range] {
                    payload_bytes.extend_from_slice(&element.to_le_bytes());
                }
            }
            AstroSsdPromptCacheHostTensorElements::Int32(elements) => {
                for element in &elements[element_range] {
                    payload_bytes.extend_from_slice(&element.to_le_bytes());
                }
            }
        }
    }

    /// The number of elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        match &self.elements {
            AstroSsdPromptCacheHostTensorElements::SixteenBit(elements) => elements.len(),
            AstroSsdPromptCacheHostTensorElements::Float32(elements) => elements.len(),
            AstroSsdPromptCacheHostTensorElements::Int32(elements) => elements.len(),
        }
    }
}

/// Total payload bytes of a block's tensors.
#[must_use]
pub fn host_tensors_payload_byte_count(host_tensors: &AstroSsdPromptCacheHostTensors) -> usize {
    host_tensors
        .values()
        .map(AstroSsdPromptCacheHostTensor::payload_byte_count)
        .sum()
}

/// Evaluates every tensor in one submission, then copies each into host
/// memory in row-major order.
///
/// # Errors
/// Fails with [`AstroSsdPromptCacheDiskStoreError::SaveSafetensors`] when MLX
/// cannot evaluate or copy a tensor, keeping the typed active-memory-limit
/// error the publication retry recognizes, and with
/// [`AstroSsdPromptCacheDiskStoreError::UnsupportedHostTensorDtype`] for a
/// dtype no decoder-state layout stores.
pub fn copy_state_tensors_to_host(
    runtime: &MlxRuntime,
    tensors: &HashMap<String, MlxArray>,
) -> Result<AstroSsdPromptCacheHostTensors, AstroSsdPromptCacheDiskStoreError> {
    let evaluated_tensors: Vec<&MlxArray> = tensors.values().collect();
    runtime
        .evaluate_arrays(&evaluated_tensors)
        .map_err(
            |source| AstroSsdPromptCacheDiskStoreError::SaveSafetensors {
                source: MlxRuntimeError::from(source),
            },
        )?;
    let mut host_tensors = AstroSsdPromptCacheHostTensors::with_capacity(tensors.len());
    for (tensor_name, tensor) in tensors {
        host_tensors.insert(
            tensor_name.clone(),
            copy_tensor_to_host(tensor_name, tensor)?,
        );
    }
    Ok(host_tensors)
}

fn copy_tensor_to_host(
    tensor_name: &str,
    tensor: &MlxArray,
) -> Result<AstroSsdPromptCacheHostTensor, AstroSsdPromptCacheDiskStoreError> {
    let shape = tensor
        .shape()
        .into_iter()
        .map(|dimension| usize::try_from(dimension).unwrap_or(0))
        .collect();
    let (safetensors_dtype_name, copied_elements) = match tensor.dtype() {
        MlxDtype::Float16 => (
            "F16",
            tensor
                .to_vec_f16_bits()
                .map(AstroSsdPromptCacheHostTensorElements::SixteenBit),
        ),
        MlxDtype::BFloat16 => (
            "BF16",
            tensor
                .to_vec_bf16_bits()
                .map(AstroSsdPromptCacheHostTensorElements::SixteenBit),
        ),
        MlxDtype::Float32 => (
            "F32",
            tensor
                .to_vec_f32()
                .map(AstroSsdPromptCacheHostTensorElements::Float32),
        ),
        MlxDtype::Int32 => (
            "I32",
            tensor
                .to_vec_i32()
                .map(AstroSsdPromptCacheHostTensorElements::Int32),
        ),
        unsupported_dtype => {
            return Err(
                AstroSsdPromptCacheDiskStoreError::UnsupportedHostTensorDtype {
                    tensor_name: tensor_name.to_owned(),
                    dtype_description: format!("{unsupported_dtype:?}"),
                },
            );
        }
    };
    let elements =
        copied_elements.map_err(
            |source| AstroSsdPromptCacheDiskStoreError::SaveSafetensors {
                source: MlxRuntimeError::from(source),
            },
        )?;
    Ok(AstroSsdPromptCacheHostTensor {
        safetensors_dtype_name,
        shape,
        elements,
    })
}

#[cfg(test)]
#[path = "host_state_tensor_tests.rs"]
mod host_state_tensor_tests;
