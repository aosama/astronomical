//! Proves one component of the 4-bit build stores exactly the tensors of its
//! BF16 counterpart, and describes each one in Diffusers terms.
//!
//! Every Diffusers tensor must be present under its mflux name, either dense
//! BF16 or as a 4-bit affine triple (`weight` U32 `[out, in / 8]`, `scales` and
//! `biases` BF16 `[out, in / 64]`). Nothing else may be stored, apart from the
//! text encoder's `rotary_emb.inv_freq`, which the engine recomputes.
//!
//! The descriptors this produces are logical: one per Diffusers tensor, named
//! and shaped as in the BF16 build, so memory geometry and admission need no
//! layout branches. Their payload is what the engine keeps resident for that
//! tensor: the packed triple for the transformer and text encoder, which
//! multiply on packed weights, and the dequantized BF16 size for the VAE,
//! which dequantizes its few attention projections at load. The offsets are
//! those of the stored `weight` tensor.

use std::collections::BTreeMap;

use ::safetensors::Dtype;

use crate::affine_quantization_scheme::{
    AFFINE_QUANTIZATION_GROUP_SIZE, AFFINE_QUANTIZED_VALUES_PER_WORD,
};
use crate::artifact_validation::{RawSafetensorsInventory, RawSafetensorsTensorDescriptor};

use super::super::artifact::Flux2KleinArtifactError;
use super::super::inventory::{Flux2KleinTensorDescriptor, public_descriptor};
use super::super::weight_layout::Flux2KleinComponent;
use super::mlx_4bit_tensor_name;
use super::shard_index::Mlx4BitShardIndex;

/// Qwen3 checkpoints carry their rotary inverse frequencies; the engine derives
/// them from the configuration instead, so the stored copy is ignored.
const IGNORED_TEXT_ENCODER_TENSOR: &str = "rotary_emb.inv_freq";
const BF16_BYTES_PER_VALUE: u64 = 2;

/// Every Diffusers tensor a component must provide, with the logical shape it
/// must have when the configuration fixes one.
pub(super) type ExpectedTensors = BTreeMap<String, Option<Vec<usize>>>;

/// One stored tensor and the shard file holding it, relative to the model
/// directory.
struct StoredTensor {
    source_file_name: String,
    descriptor: RawSafetensorsTensorDescriptor,
}

/// Validates the component's shards against `expected_tensors` and returns its
/// logical descriptors in Diffusers names.
pub(super) fn validate_component_tensors(
    component: Flux2KleinComponent,
    expected_tensors: &ExpectedTensors,
    index: &Mlx4BitShardIndex,
    shard_inventories: Vec<(String, RawSafetensorsInventory)>,
) -> Result<Vec<Flux2KleinTensorDescriptor>, Flux2KleinArtifactError> {
    let mut stored_tensors = collect_stored_tensors(component, index, shard_inventories)?;
    if component == Flux2KleinComponent::TextEncoder {
        stored_tensors.remove(IGNORED_TEXT_ENCODER_TENSOR);
    }
    let mut descriptors = Vec::with_capacity(expected_tensors.len());
    for (diffusers_name, expected_shape) in expected_tensors {
        let stored_name = mlx_4bit_tensor_name(component, diffusers_name);
        let descriptor = take_logical_tensor(
            component,
            &mut stored_tensors,
            &stored_name,
            diffusers_name,
            expected_shape.as_deref(),
        )?;
        descriptors.push(descriptor);
    }
    if let Some(unexpected_name) = stored_tensors.keys().next() {
        return Err(Flux2KleinArtifactError::UnsupportedTensor {
            component: component.label(),
            tensor_name: unexpected_name.clone(),
        });
    }
    Ok(descriptors)
}

/// Every stored tensor by name, after proving the shard index and the shard
/// headers agree on where each tensor lives.
fn collect_stored_tensors(
    component: Flux2KleinComponent,
    index: &Mlx4BitShardIndex,
    shard_inventories: Vec<(String, RawSafetensorsInventory)>,
) -> Result<BTreeMap<String, StoredTensor>, Flux2KleinArtifactError> {
    let mut stored_tensors = BTreeMap::new();
    for (shard_file_name, shard_inventory) in shard_inventories {
        let source_file_name = format!("{}/{shard_file_name}", component.folder_name());
        for descriptor in shard_inventory.tensor_descriptors {
            let indexed_shard = index.weight_map.get(&descriptor.tensor_name);
            if indexed_shard != Some(&shard_file_name)
                || stored_tensors.contains_key(&descriptor.tensor_name)
            {
                return Err(shard_index_disagreement(component, descriptor.tensor_name));
            }
            stored_tensors.insert(
                descriptor.tensor_name.clone(),
                StoredTensor {
                    source_file_name: source_file_name.clone(),
                    descriptor,
                },
            );
        }
    }
    if let Some(unstored_name) = index
        .weight_map
        .keys()
        .find(|indexed_name| !stored_tensors.contains_key(*indexed_name))
    {
        return Err(shard_index_disagreement(component, unstored_name.clone()));
    }
    Ok(stored_tensors)
}

/// Removes the stored tensors behind `diffusers_name` and describes them as
/// that one logical tensor.
fn take_logical_tensor(
    component: Flux2KleinComponent,
    stored_tensors: &mut BTreeMap<String, StoredTensor>,
    stored_name: &str,
    diffusers_name: &str,
    expected_shape: Option<&[usize]>,
) -> Result<Flux2KleinTensorDescriptor, Flux2KleinArtifactError> {
    let weight = stored_tensors.remove(stored_name).ok_or_else(|| {
        Flux2KleinArtifactError::MissingTensor {
            component: component.label(),
            tensor_name: diffusers_name.to_owned(),
        }
    })?;
    let (scales, biases) = match stored_name.strip_suffix(".weight") {
        Some(stem) => (
            stored_tensors.remove(&format!("{stem}.scales")),
            stored_tensors.remove(&format!("{stem}.biases")),
        ),
        None => (None, None),
    };
    let logical_tensor = match (scales, biases) {
        (None, None) => dense_logical_tensor(component, weight, diffusers_name)?,
        (Some(scales), Some(biases)) => {
            quantized_logical_tensor(component, weight, &scales, &biases, diffusers_name)?
        }
        _ => return Err(packing_error(component, diffusers_name)),
    };
    if expected_shape.is_some_and(|shape| shape != logical_tensor.descriptor.shape.as_slice()) {
        return Err(Flux2KleinArtifactError::TensorShape {
            component: component.label(),
            tensor_name: diffusers_name.to_owned(),
        });
    }
    Ok(public_descriptor(
        &logical_tensor.source_file_name,
        logical_tensor.descriptor,
    ))
}

/// A dense BF16 tensor. A VAE convolution kernel is stored in MLX layout
/// `[out, kh, kw, in]` and described in the Diffusers layout `[out, in, kh, kw]`.
fn dense_logical_tensor(
    component: Flux2KleinComponent,
    mut stored: StoredTensor,
    diffusers_name: &str,
) -> Result<StoredTensor, Flux2KleinArtifactError> {
    if stored.descriptor.dtype != Dtype::BF16 {
        return Err(Flux2KleinArtifactError::TensorDtype {
            component: component.label(),
            tensor_name: diffusers_name.to_owned(),
        });
    }
    if let (Flux2KleinComponent::Vae, &[output, kernel_height, kernel_width, input]) =
        (component, stored.descriptor.shape.as_slice())
    {
        stored.descriptor.shape = vec![output, input, kernel_height, kernel_width];
    }
    diffusers_name.clone_into(&mut stored.descriptor.tensor_name);
    Ok(stored)
}

/// A 4-bit affine triple, described as the `[out, in]` projection it encodes.
fn quantized_logical_tensor(
    component: Flux2KleinComponent,
    mut weight: StoredTensor,
    scales: &StoredTensor,
    biases: &StoredTensor,
    diffusers_name: &str,
) -> Result<StoredTensor, Flux2KleinArtifactError> {
    let [output_width, input_width] =
        packed_projection_widths(&weight.descriptor, &scales.descriptor, &biases.descriptor)
            .ok_or_else(|| packing_error(component, diffusers_name))?;
    let dequantizes_at_load = component == Flux2KleinComponent::Vae;
    let resident_payload_bytes = if !dequantizes_at_load {
        checked_payload_sum(&[
            weight.descriptor.tensor_payload_bytes,
            scales.descriptor.tensor_payload_bytes,
            biases.descriptor.tensor_payload_bytes,
        ])
    } else {
        dequantized_payload_bytes(output_width, input_width)
    }
    .ok_or(Flux2KleinArtifactError::PayloadAccountingOverflow)?;
    if dequantizes_at_load {
        weight.descriptor.dtype = Dtype::BF16;
    }
    weight.descriptor.shape = vec![output_width, input_width];
    weight.descriptor.tensor_payload_bytes = resident_payload_bytes;
    diffusers_name.clone_into(&mut weight.descriptor.tensor_name);
    Ok(weight)
}

/// The logical `[out, in]` of a correctly packed triple, or `None`.
fn packed_projection_widths(
    weight: &RawSafetensorsTensorDescriptor,
    scales: &RawSafetensorsTensorDescriptor,
    biases: &RawSafetensorsTensorDescriptor,
) -> Option<[usize; 2]> {
    let (&[output_width, packed_width], &[scale_rows, group_count]) =
        (weight.shape.as_slice(), scales.shape.as_slice())
    else {
        return None;
    };
    let input_width = packed_width.checked_mul(AFFINE_QUANTIZED_VALUES_PER_WORD)?;
    let is_packed = weight.dtype == Dtype::U32
        && scales.dtype == Dtype::BF16
        && biases.dtype == Dtype::BF16
        && biases.shape == scales.shape
        && scale_rows == output_width
        && group_count.checked_mul(AFFINE_QUANTIZATION_GROUP_SIZE) == Some(input_width)
        && input_width > 0;
    is_packed.then_some([output_width, input_width])
}

fn checked_payload_sum(payload_bytes: &[u64]) -> Option<u64> {
    let mut total_payload_bytes = 0_u64;
    for tensor_payload_bytes in payload_bytes {
        total_payload_bytes = total_payload_bytes.checked_add(*tensor_payload_bytes)?;
    }
    Some(total_payload_bytes)
}

fn dequantized_payload_bytes(output_width: usize, input_width: usize) -> Option<u64> {
    u64::try_from(output_width)
        .ok()?
        .checked_mul(u64::try_from(input_width).ok()?)?
        .checked_mul(BF16_BYTES_PER_VALUE)
}

fn packing_error(component: Flux2KleinComponent, diffusers_name: &str) -> Flux2KleinArtifactError {
    Flux2KleinArtifactError::QuantizedTensorPacking {
        component: component.label(),
        tensor_name: diffusers_name.to_owned(),
    }
}

fn shard_index_disagreement(
    component: Flux2KleinComponent,
    tensor_name: String,
) -> Flux2KleinArtifactError {
    Flux2KleinArtifactError::ShardIndexDisagreement {
        component: component.label(),
        tensor_name,
    }
}

#[cfg(test)]
#[path = "tensor_profile_tests.rs"]
mod tests;
