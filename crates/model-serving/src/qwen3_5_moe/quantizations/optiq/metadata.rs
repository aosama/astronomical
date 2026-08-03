use std::collections::BTreeMap;

use serde::Deserialize;
use thiserror::Error;

use crate::qwen3_5_moe::Qwen3_5MoEConfig;

use super::config::{
    OptiQQuantizationProfile, is_mlx_affine_quantization_bit_width_supported,
    is_mlx_affine_quantization_group_size_supported,
};

const MAXIMUM_OPTIQ_METADATA_BYTES: usize = 128 * 1024;
/// Strict metadata for the measured portion of the pinned OptiQ bit map.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptiQMetadata {
    measured_module_profiles: BTreeMap<String, OptiQQuantizationProfile>,
}

impl OptiQMetadata {
    /// Parses and validates the complete bounded OptiQ metadata document.
    pub fn from_json_bytes(metadata_bytes: &[u8]) -> Result<Self, OptiQMetadataError> {
        if metadata_bytes.len() > MAXIMUM_OPTIQ_METADATA_BYTES {
            return Err(OptiQMetadataError::MetadataTooLarge {
                actual_size_bytes: metadata_bytes.len(),
                maximum_size_bytes: MAXIMUM_OPTIQ_METADATA_BYTES,
            });
        }
        let metadata_document = serde_json::from_slice::<OptiQMetadataDocument>(metadata_bytes)
            .map_err(OptiQMetadataError::DeserializeMetadata)?;
        let mut measured_module_profiles = BTreeMap::new();
        for (module_name, quantization_override) in metadata_document.per_layer {
            if !is_mlx_affine_quantization_group_size_supported(quantization_override.group_size) {
                return Err(OptiQMetadataError::UnsupportedGroupSize {
                    module_name,
                    actual_group_size: quantization_override.group_size,
                });
            }
            if !is_mlx_affine_quantization_bit_width_supported(quantization_override.bits) {
                return Err(OptiQMetadataError::UnsupportedBits {
                    module_name,
                    actual_bits: quantization_override.bits,
                });
            }
            measured_module_profiles.insert(
                module_name,
                OptiQQuantizationProfile {
                    bits: quantization_override.bits,
                    group_size: quantization_override.group_size,
                },
            );
        }
        Ok(Self {
            measured_module_profiles,
        })
    }

    /// Returns the number of sensitivity-measured quantized modules.
    #[must_use]
    pub fn measured_module_count(&self) -> usize {
        self.measured_module_profiles.len()
    }

    /// Requires every declared measured module profile to equal a config override.
    pub fn validate_against_config(
        &self,
        qwen3_5_moe_config: &Qwen3_5MoEConfig,
    ) -> Result<(), OptiQMetadataError> {
        let expected_measured_module_profiles: BTreeMap<String, OptiQQuantizationProfile> =
            qwen3_5_moe_config
                .quantized_module_profiles()
                .iter()
                .filter(|(_, profile)| !profile.is_unquantized())
                .map(|(name, profile)| (name.clone(), *profile))
                .collect();
        for (module_name, actual_profile) in &self.measured_module_profiles {
            let expected_profile = match expected_measured_module_profiles.get(module_name) {
                Some(expected_profile) => expected_profile,
                None => {
                    return Err(OptiQMetadataError::UnexpectedMeasuredModule {
                        module_name: module_name.clone(),
                    });
                }
            };
            if actual_profile.bits != expected_profile.bits {
                return Err(OptiQMetadataError::ConfigBitMismatch {
                    module_name: module_name.clone(),
                    config_bits: expected_profile.bits,
                    metadata_bits: actual_profile.bits,
                });
            }
            if actual_profile.group_size != expected_profile.group_size {
                return Err(OptiQMetadataError::ConfigGroupSizeMismatch {
                    module_name: module_name.clone(),
                    config_group_size: expected_profile.group_size,
                    metadata_group_size: actual_profile.group_size,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptiQMetadataDocument {
    #[allow(dead_code)]
    method: String,
    #[allow(dead_code)]
    base_model: String,
    #[allow(dead_code)]
    reference: String,
    #[serde(default)]
    #[allow(dead_code)]
    sensitivity_measured_on: Option<String>,
    #[allow(dead_code)]
    target_bpw: f64,
    #[allow(dead_code)]
    achieved_bpw: f64,
    #[allow(dead_code)]
    n_high_bits: usize,
    #[allow(dead_code)]
    n_low_bits: usize,
    #[allow(dead_code)]
    threshold: f64,
    per_layer: BTreeMap<String, OptiQQuantizationOverride>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct OptiQQuantizationOverride {
    bits: u32,
    group_size: u32,
}

/// A strict mismatch in the pinned OptiQ sensitivity metadata.
#[derive(Debug, Error)]
pub enum OptiQMetadataError {
    #[error("OptiQ metadata is {actual_size_bytes} bytes, exceeding {maximum_size_bytes}")]
    MetadataTooLarge {
        actual_size_bytes: usize,
        maximum_size_bytes: usize,
    },
    #[error("failed to decode OptiQ metadata JSON")]
    DeserializeMetadata(#[source] serde_json::Error),
    #[error(
        "OptiQ metadata module '{module_name}' uses unsupported group size {actual_group_size}"
    )]
    UnsupportedGroupSize {
        module_name: String,
        actual_group_size: u32,
    },
    #[error(
        "OptiQ metadata module '{module_name}' uses unsupported {actual_bits}-bit quantization"
    )]
    UnsupportedBits {
        module_name: String,
        actual_bits: u32,
    },
    #[error(
        "OptiQ module '{module_name}' is {config_bits}-bit in config and {metadata_bits}-bit in metadata"
    )]
    ConfigBitMismatch {
        module_name: String,
        config_bits: u32,
        metadata_bits: u32,
    },
    #[error(
        "OptiQ module '{module_name}' has group size {config_group_size} in config and {metadata_group_size} in metadata"
    )]
    ConfigGroupSizeMismatch {
        module_name: String,
        config_group_size: u32,
        metadata_group_size: u32,
    },
    #[error("OptiQ metadata contains unexpected measured module '{module_name}'")]
    UnexpectedMeasuredModule { module_name: String },
}
