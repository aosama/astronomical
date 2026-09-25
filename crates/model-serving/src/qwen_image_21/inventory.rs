//! Physical tensor inventory validation for the three Qwen-Image-2.1 weight components.
//!
//! Each component validates its retained safetensors header against the exact expected profile
//! (name, dtype, shape) produced from the validated component config: a package passes only if
//! its physical tensors are exactly the reviewed set — no missing, no extra, no re-shape.

use std::collections::{BTreeMap, BTreeSet};

use ::safetensors::Dtype;

use crate::artifact_validation::RawSafetensorsInventory;

use super::artifact::QwenImage21ArtifactError;
use super::configuration::{
    QwenImage21TextEncoderConfig, QwenImage21TransformerConfig, QwenImage21VaeConfig,
};
use super::tensor_profiles::{
    QwenImage21TensorProfile, text_encoder_tensor_profiles, transformer_tensor_profiles,
    vae_tensor_profiles,
};

/// One exact physical tensor source interval retained for native loading.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21TensorDescriptor {
    source_file_name: String,
    tensor_name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data_start_offset_bytes: u64,
    data_end_offset_bytes: u64,
    payload_bytes: u64,
}

impl QwenImage21TensorDescriptor {
    pub fn source_file_name(&self) -> &str {
        &self.source_file_name
    }
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub const fn data_start_offset_bytes(&self) -> u64 {
        self.data_start_offset_bytes
    }
    pub const fn data_end_offset_bytes(&self) -> u64 {
        self.data_end_offset_bytes
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

/// Validated physical inventory of one weight component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenImage21TensorInventory {
    descriptors: Vec<QwenImage21TensorDescriptor>,
    payload_bytes: u64,
}

impl QwenImage21TensorInventory {
    pub fn descriptors(&self) -> &[QwenImage21TensorDescriptor] {
        &self.descriptors
    }
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub fn tensor_count(&self) -> usize {
        self.descriptors.len()
    }
}

fn expected_dtype(
    component: &'static str,
    tensor_name: &str,
    declared_dtype: &str,
) -> Result<Dtype, QwenImage21ArtifactError> {
    match declared_dtype {
        "U32" => Ok(Dtype::U32),
        "BF16" => Ok(Dtype::BF16),
        "F32" => Ok(Dtype::F32),
        unsupported => Err(QwenImage21ArtifactError::UnsupportedProfileDtype {
            component,
            tensor_name: tensor_name.to_owned(),
            declared_dtype: unsupported.to_owned(),
        }),
    }
}

fn validate_component(
    component: &'static str,
    file_name: &str,
    inventory: RawSafetensorsInventory,
    expected: Vec<QwenImage21TensorProfile>,
) -> Result<QwenImage21TensorInventory, QwenImage21ArtifactError> {
    let expected_by_name = expected
        .iter()
        .map(|profile| (profile.tensor_name.as_str(), profile))
        .collect::<BTreeMap<_, _>>();
    let actual_names = inventory
        .tensor_descriptors
        .iter()
        .map(|descriptor| descriptor.tensor_name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(missing_name) = expected_by_name
        .keys()
        .find(|name| !actual_names.contains(*name))
    {
        return Err(QwenImage21ArtifactError::MissingTensor {
            component,
            tensor_name: String::from(*missing_name),
        });
    }
    if let Some(extra_name) = actual_names
        .iter()
        .find(|name| !expected_by_name.contains_key(**name))
    {
        return Err(QwenImage21ArtifactError::UnsupportedTensor {
            component,
            tensor_name: String::from(*extra_name),
        });
    }
    let descriptors = inventory
        .tensor_descriptors
        .into_iter()
        .map(|descriptor| {
            // Unreachable after the exact-name checks above; reporting the missing tensor
            // again beats panicking if that invariant ever regresses.
            let expected_profile = expected_by_name
                .get(descriptor.tensor_name.as_str())
                .ok_or_else(|| QwenImage21ArtifactError::MissingTensor {
                    component,
                    tensor_name: descriptor.tensor_name.clone(),
                })?;
            if descriptor.dtype
                != expected_dtype(component, &descriptor.tensor_name, expected_profile.dtype)?
            {
                return Err(QwenImage21ArtifactError::TensorDtype {
                    component,
                    tensor_name: descriptor.tensor_name,
                });
            }
            if descriptor.shape != expected_profile.shape {
                return Err(QwenImage21ArtifactError::TensorShape {
                    component,
                    tensor_name: descriptor.tensor_name,
                });
            }
            Ok(QwenImage21TensorDescriptor {
                source_file_name: file_name.to_owned(),
                tensor_name: descriptor.tensor_name,
                dtype: descriptor.dtype,
                shape: descriptor.shape,
                data_start_offset_bytes: descriptor.data_start_offset_bytes,
                data_end_offset_bytes: descriptor.data_end_offset_bytes,
                payload_bytes: descriptor.tensor_payload_bytes,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(QwenImage21TensorInventory {
        payload_bytes: inventory.shard_payload_bytes,
        descriptors,
    })
}

pub(super) fn validate_transformer_inventory(
    file_name: &str,
    inventory: RawSafetensorsInventory,
    config: &QwenImage21TransformerConfig,
) -> Result<QwenImage21TensorInventory, QwenImage21ArtifactError> {
    let expected = transformer_tensor_profiles(config);
    validate_component("transformer", file_name, inventory, expected)
}

pub(super) fn validate_text_encoder_inventory(
    file_name: &str,
    inventory: RawSafetensorsInventory,
    config: &QwenImage21TextEncoderConfig,
) -> Result<QwenImage21TensorInventory, QwenImage21ArtifactError> {
    let expected = text_encoder_tensor_profiles(config);
    validate_component("text_encoder", file_name, inventory, expected)
}

pub(super) fn validate_vae_inventory(
    file_name: &str,
    inventory: RawSafetensorsInventory,
    config: &QwenImage21VaeConfig,
) -> Result<QwenImage21TensorInventory, QwenImage21ArtifactError> {
    let expected = vae_tensor_profiles(config);
    validate_component("vae", file_name, inventory, expected)
}
