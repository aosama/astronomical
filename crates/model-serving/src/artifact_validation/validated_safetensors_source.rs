use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::safetensors_dtype::dtype_bits_per_element;
use super::{
    ArtifactValidationError, RequiredFileProfile, TensorDtype, TensorFeature, TensorInventory,
    TensorLocation, TensorProfile, TensorSourceId, ValidatedRequiredFile, ValidatedWeightsFile,
    validate_required_file,
};
use crate::safetensors::{
    BoundedSafetensorsHeaderError, SafetensorsTensorView, read_bounded_safetensors_json_header,
};

const MAXIMUM_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;

/// Already-open validated SafeTensors descriptor with one retained bounded header.
#[derive(Debug)]
pub(crate) struct ValidatedSafetensorsSource {
    source_id: TensorSourceId,
    required_file: ValidatedRequiredFile,
    tensor_metadata_by_stored_name: BTreeMap<String, SafetensorsTensorView>,
    payload_bytes: u64,
}

/// Exercises the required/optional profile partition through the real descriptor and header path.
///
/// The outer `Result` represents required source validity. The boolean is `false` only when the
/// requested optional feature has a profile defect and therefore must be disabled atomically.
#[doc(hidden)]
pub fn validate_safetensors_profile_partitions_for_tests(
    model_directory: &Path,
    relative_file_name: &str,
    inventory: &TensorInventory,
    canonical_profiles: &[TensorProfile],
    optional_feature: TensorFeature,
) -> Result<bool, ArtifactValidationError> {
    let required_file = validate_required_file(
        model_directory,
        &RequiredFileProfile {
            file_name: relative_file_name.to_owned(),
            size_bytes: 0,
        },
    )?;
    let source = ValidatedSafetensorsSource::parse(TensorSourceId::new(1), required_file)?;
    source.validate_required_inventory_profiles(inventory, canonical_profiles)?;
    Ok(source
        .validate_feature_inventory_profiles(inventory, canonical_profiles, optional_feature)
        .is_ok())
}

impl ValidatedSafetensorsSource {
    pub(crate) fn parse(
        source_id: TensorSourceId,
        required_file: ValidatedRequiredFile,
    ) -> Result<Self, ArtifactValidationError> {
        let bounded_header = read_bounded_safetensors_json_header(
            required_file.file(),
            required_file.size_bytes(),
            MAXIMUM_SAFETENSORS_HEADER_BYTES,
        )
        .map_err(|error| map_header_error(error, required_file.file_name()))?;
        if let Some(metadata_json_value) = bounded_header.metadata_json_value {
            serde_json::from_value::<BTreeMap<String, String>>(metadata_json_value).map_err(
                |source| ArtifactValidationError::InvalidSafetensorsHeader {
                    file_name: required_file.file_name().to_owned(),
                    source,
                },
            )?;
        }
        let mut tensor_metadata_by_stored_name = BTreeMap::new();
        for (stored_name, tensor_json_value) in bounded_header.tensor_json_values {
            let tensor_metadata = serde_json::from_value::<SafetensorsTensorView>(
                tensor_json_value,
            )
            .map_err(|source| ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: required_file.file_name().to_owned(),
                source,
            })?;
            tensor_metadata_by_stored_name.insert(stored_name, tensor_metadata);
        }
        let payload_bytes = validate_physical_metadata(
            &tensor_metadata_by_stored_name,
            bounded_header.data_section_start_bytes,
            bounded_header.file_size_bytes,
            required_file.file_name(),
        )?;
        Ok(Self {
            source_id,
            required_file,
            tensor_metadata_by_stored_name,
            payload_bytes,
        })
    }

    pub(crate) const fn source_id(&self) -> TensorSourceId {
        self.source_id
    }

    pub(crate) fn file_name(&self) -> &str {
        self.required_file.file_name()
    }

    pub(crate) const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub(crate) fn stored_tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensor_metadata_by_stored_name.keys()
    }

    /// Validates canonical profiles against physical names without reparsing the header.
    pub(crate) fn validate_inventory_profiles(
        &self,
        inventory: &TensorInventory,
        canonical_profiles: &[TensorProfile],
    ) -> Result<(), ArtifactValidationError> {
        let source_locations = self.source_locations(inventory);
        self.validate_exact_physical_inventory(&source_locations)?;
        self.validate_locations(&source_locations, canonical_profiles)
    }

    /// Validates required target and vision profiles while leaving optional features atomic.
    ///
    /// A target shard may physically contain an optional MTP head. A wrong optional dtype or
    /// shape must disable that complete feature, not reject otherwise valid target weights.
    /// Physical-name and offset validation still covers the entire source before this split, so
    /// ignoring optional profile semantics cannot hide an undeclared or structurally unsafe tensor.
    pub(crate) fn validate_required_inventory_profiles(
        &self,
        inventory: &TensorInventory,
        canonical_profiles: &[TensorProfile],
    ) -> Result<(), ArtifactValidationError> {
        let source_locations = self.source_locations(inventory);
        self.validate_exact_physical_inventory(&source_locations)?;
        let required_locations = source_locations
            .into_iter()
            .filter(|location| location.feature().is_none())
            .collect::<Vec<_>>();
        self.validate_locations(&required_locations, canonical_profiles)
    }

    /// Validates one optional feature independently after required profiles are known safe.
    pub(crate) fn validate_feature_inventory_profiles(
        &self,
        inventory: &TensorInventory,
        canonical_profiles: &[TensorProfile],
        feature: TensorFeature,
    ) -> Result<(), ArtifactValidationError> {
        let feature_locations = self
            .source_locations(inventory)
            .into_iter()
            .filter(|location| location.feature() == Some(feature))
            .collect::<Vec<_>>();
        self.validate_locations(&feature_locations, canonical_profiles)
    }

    fn source_locations<'inventory>(
        &self,
        inventory: &'inventory TensorInventory,
    ) -> Vec<&'inventory TensorLocation> {
        inventory
            .locations()
            .filter(|location| location.source_id() == self.source_id)
            .collect()
    }

    fn validate_exact_physical_inventory(
        &self,
        source_locations: &[&TensorLocation],
    ) -> Result<(), ArtifactValidationError> {
        let declared_stored_names = source_locations
            .iter()
            .map(|location| location.stored_name())
            .collect::<BTreeSet<_>>();
        let physical_stored_names = self
            .tensor_metadata_by_stored_name
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if declared_stored_names == physical_stored_names {
            return Ok(());
        }
        let unresolved_stored_name = physical_stored_names
            .difference(&declared_stored_names)
            .next()
            .or_else(|| {
                declared_stored_names
                    .difference(&physical_stored_names)
                    .next()
            })
            .copied()
            .unwrap_or("unresolved tensor inventory");
        Err(ArtifactValidationError::UnexpectedTensor {
            tensor_name: unresolved_stored_name.to_owned(),
        })
    }

    fn validate_locations(
        &self,
        locations: &[&TensorLocation],
        canonical_profiles: &[TensorProfile],
    ) -> Result<(), ArtifactValidationError> {
        let profile_by_canonical_name = canonical_profiles
            .iter()
            .map(|profile| (profile.name.as_str(), profile))
            .collect::<BTreeMap<_, _>>();
        for location in locations {
            let profile = profile_by_canonical_name
                .get(location.canonical_name())
                .ok_or_else(|| ArtifactValidationError::TensorMissing {
                    tensor_name: location.canonical_name().to_owned(),
                    file_name: self.file_name().to_owned(),
                })?;
            let metadata = self
                .tensor_metadata_by_stored_name
                .get(location.stored_name())
                .ok_or_else(|| ArtifactValidationError::TensorMissing {
                    tensor_name: location.canonical_name().to_owned(),
                    file_name: self.file_name().to_owned(),
                })?;
            validate_profile(profile, metadata)?;
        }
        Ok(())
    }

    pub(crate) fn into_validated_weights_file(
        self,
    ) -> Result<ValidatedWeightsFile, ArtifactValidationError> {
        self.required_file.into_validated_weights_file()
    }
}

fn validate_physical_metadata(
    tensors: &BTreeMap<String, SafetensorsTensorView>,
    data_section_start_bytes: u64,
    file_size_bytes: u64,
    file_name: &str,
) -> Result<u64, ArtifactValidationError> {
    let mut ordered_tensors = tensors.iter().collect::<Vec<_>>();
    ordered_tensors.sort_by_key(|(_, metadata)| metadata.data_start_offset());
    let mut expected_payload_offset = 0_u64;
    for (stored_name, metadata) in ordered_tensors {
        let element_count = metadata
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(
                        u64::try_from(*dimension)
                            .map_err(|_| ArtifactValidationError::TensorPayloadSizeOverflow)?,
                    )
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)
            })?;
        let expected_bits = element_count
            .checked_mul(dtype_bits_per_element(
                &metadata.dtype,
                file_name,
                stored_name,
            )?)
            .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
        // SafeTensors requires every tensor payload to end on a complete byte.
        if expected_bits % 8 != 0 {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: file_name.to_owned(),
                tensor_name: stored_name.clone(),
                data_start_offset: metadata.data_start_offset(),
                data_end_offset: metadata.data_end_offset(),
            });
        }
        let expected_bytes = expected_bits / 8;
        if metadata.data_start_offset() != expected_payload_offset
            || metadata
                .data_end_offset()
                .checked_sub(metadata.data_start_offset())
                != Some(expected_bytes)
        {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: file_name.to_owned(),
                tensor_name: stored_name.clone(),
                data_start_offset: metadata.data_start_offset(),
                data_end_offset: metadata.data_end_offset(),
            });
        }
        expected_payload_offset = metadata.data_end_offset();
    }
    let actual_payload_bytes = file_size_bytes
        .checked_sub(data_section_start_bytes)
        .ok_or(ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: file_name.to_owned(),
            expected_minimum_bytes: data_section_start_bytes,
            actual_file_size_bytes: file_size_bytes,
        })?;
    if expected_payload_offset != actual_payload_bytes {
        return Err(ArtifactValidationError::SafetensorsPayloadLengthMismatch {
            file_name: file_name.to_owned(),
            declared_payload_bytes: expected_payload_offset,
            actual_payload_bytes,
        });
    }
    Ok(actual_payload_bytes)
}

fn validate_profile(
    profile: &TensorProfile,
    metadata: &SafetensorsTensorView,
) -> Result<(), ArtifactValidationError> {
    let dtype_matches = match profile.dtype {
        TensorDtype::AffineQuantizationFloat | TensorDtype::ModelFloat => {
            matches!(metadata.dtype.as_str(), "F16" | "BF16" | "F32")
        }
        TensorDtype::BFloat16 => metadata.dtype == "BF16",
        TensorDtype::Float32 => metadata.dtype == "F32",
        TensorDtype::UInt32 => metadata.dtype == "U32",
    };
    if !dtype_matches {
        return Err(ArtifactValidationError::UnexpectedTensor {
            tensor_name: profile.name.clone(),
        });
    }
    if metadata.shape != profile.shape {
        return Err(ArtifactValidationError::TensorShapeMismatch {
            tensor_name: profile.name.clone(),
            expected_shape: profile.shape.clone(),
            actual_shape: metadata.shape.clone(),
        });
    }
    Ok(())
}

fn map_header_error(
    error: BoundedSafetensorsHeaderError,
    file_name: &str,
) -> ArtifactValidationError {
    match error {
        BoundedSafetensorsHeaderError::ReadLengthPrefix(source) => {
            ArtifactValidationError::ReadSafetensorsLengthPrefix {
                file_name: file_name.to_owned(),
                source,
            }
        }
        BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        } => ArtifactValidationError::SafetensorsHeaderLengthTooLarge {
            file_name: file_name.to_owned(),
            header_length_bytes,
            maximum_header_length_bytes,
        },
        BoundedSafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset_bytes,
            file_size_bytes,
        } => ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: file_name.to_owned(),
            expected_minimum_bytes: header_end_offset_bytes,
            actual_file_size_bytes: file_size_bytes,
        },
        BoundedSafetensorsHeaderError::ReadHeader(source) => {
            ArtifactValidationError::ReadSafetensorsHeader {
                file_name: file_name.to_owned(),
                source,
            }
        }
        BoundedSafetensorsHeaderError::InvalidHeaderJson(source) => {
            ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: file_name.to_owned(),
                source,
            }
        }
    }
}
