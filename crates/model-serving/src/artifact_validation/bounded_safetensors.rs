//! Bounded safetensors header reader that never allocates the entire weights file.
//!
//! The safetensors format is: 8-byte little-endian u64 header length, then the JSON
//! header bytes, then the tensor payload data. This module reads only the length
//! prefix and the bounded header, validates every tensor offset against the open
//! file size, and returns parsed metadata without touching the multi-hundred-megabyte
//! payload region.

use std::collections::{HashMap, HashSet};
use std::fs::File;

use super::safetensors_dtype::{dtype_bits_per_element, parse_safetensors_dtype};
use super::{ArtifactValidationError, TensorDtype, TensorProfile};
use crate::safetensors::{
    BoundedSafetensorsHeaderError, SafetensorsTensorView, read_bounded_safetensors_json_header,
};

const MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES: u64 = 16 * 1024 * 1024;

struct ArtifactSafetensorsHeader {
    tensors: HashMap<String, SafetensorsTensorView>,
    data_section_start_bytes: u64,
}

/// Parsed safetensors metadata extracted from a bounded header read.
#[derive(Debug)]
pub(crate) struct BoundedSafetensorsMetadata {
    /// Total tensor payload bytes across all validated tensors.
    pub(crate) total_payload_bytes: u64,
}

/// Parsed metadata returned by partial-profile safetensors validation.
#[derive(Debug)]
pub struct PartialProfileMetadata {
    /// Total tensor payload bytes across all tensors in the shard.
    pub total_payload_bytes: u64,
}

/// Validates a safetensors shard where some tensors have strict dtype/shape profiles
/// and the remaining tensors are accepted by name only.
///
/// Qwen3.5-MoE uses exact dtype and shape profiles generated from its validated config.
pub fn validate_bounded_safetensors_with_partial_profiles(
    weights_file: &File,
    file_size_bytes: u64,
    weights_file_name: &str,
    profiled_tensor_profiles: &[TensorProfile],
    accepted_extra_tensor_names: &HashSet<&str>,
) -> Result<PartialProfileMetadata, ArtifactValidationError> {
    let mut expected_tensor_names = profiled_tensor_profiles
        .iter()
        .map(|tensor_profile| tensor_profile.name.as_str())
        .collect::<HashSet<_>>();
    expected_tensor_names.extend(accepted_extra_tensor_names.iter().copied());
    let bounded_metadata = validate_bounded_safetensors_internal(
        weights_file,
        file_size_bytes,
        weights_file_name,
        &expected_tensor_names,
        profiled_tensor_profiles,
    )?;

    Ok(PartialProfileMetadata {
        total_payload_bytes: bounded_metadata.total_payload_bytes,
    })
}

/// Validates a safetensors shard where profiled tensors have strict dtype/shape checks
/// and ALL other tensors in the shard are accepted without profiling.
///
/// This is used for models with embedded vision tensors where the vision tensors
/// are distributed across language shards but don't have language profiles.
/// Unlike `validate_bounded_safetensors_with_partial_profiles`, this function
/// does NOT require an explicit set of accepted extra tensor names — any tensor
/// in the shard that is not in `profiled_tensor_profiles` is accepted as-is.
/// It also does NOT require all profiled tensors to be present in the shard,
/// since profiled tensors may be distributed across different shards.
pub fn validate_bounded_safetensors_with_permissive_extras(
    weights_file: &File,
    file_size_bytes: u64,
    weights_file_name: &str,
    profiled_tensor_profiles: &[TensorProfile],
) -> Result<PartialProfileMetadata, ArtifactValidationError> {
    let bounded_metadata = validate_bounded_safetensors_internal_permissive(
        weights_file,
        file_size_bytes,
        weights_file_name,
        profiled_tensor_profiles,
    )?;

    Ok(PartialProfileMetadata {
        total_payload_bytes: bounded_metadata.total_payload_bytes,
    })
}

fn validate_bounded_safetensors_internal(
    weights_file: &File,
    file_size_bytes: u64,
    weights_file_name: &str,
    expected_tensor_names: &HashSet<&str>,
    tensor_profiles: &[TensorProfile],
) -> Result<BoundedSafetensorsMetadata, ArtifactValidationError> {
    let artifact_safetensors_header =
        read_artifact_safetensors_header(weights_file, file_size_bytes, weights_file_name)?;
    let bounded_metadata = validate_all_tensors(
        &artifact_safetensors_header.tensors,
        expected_tensor_names,
        tensor_profiles,
        artifact_safetensors_header.data_section_start_bytes,
        file_size_bytes,
        weights_file_name,
    )?;
    let actual_payload_bytes = file_size_bytes
        .checked_sub(artifact_safetensors_header.data_section_start_bytes)
        .ok_or(ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: weights_file_name.to_owned(),
            expected_minimum_bytes: artifact_safetensors_header.data_section_start_bytes,
            actual_file_size_bytes: file_size_bytes,
        })?;
    if bounded_metadata.total_payload_bytes != actual_payload_bytes {
        return Err(ArtifactValidationError::SafetensorsPayloadLengthMismatch {
            file_name: weights_file_name.to_owned(),
            declared_payload_bytes: bounded_metadata.total_payload_bytes,
            actual_payload_bytes,
        });
    }
    Ok(bounded_metadata)
}

/// Validates that tensor data offsets are contiguous starting from zero and that
/// each tensor's data range matches its declared shape and dtype byte size.
fn validate_tensor_data_consistency(
    safetensors_tensors: &HashMap<String, SafetensorsTensorView>,
    weights_file_name: &str,
) -> Result<(), ArtifactValidationError> {
    let mut ordered_tensors: Vec<(&String, &SafetensorsTensorView)> =
        safetensors_tensors.iter().collect();
    ordered_tensors.sort_by_key(|(_, tensor_view)| tensor_view.data_start_offset());

    let mut expected_start_offset: u64 = 0;
    for (tensor_name, tensor_view) in &ordered_tensors {
        if tensor_view.data_start_offset() != expected_start_offset {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: weights_file_name.to_owned(),
                tensor_name: (*tensor_name).clone(),
                data_start_offset: tensor_view.data_start_offset(),
                data_end_offset: tensor_view.data_end_offset(),
            });
        }
        if tensor_view.data_end_offset() < tensor_view.data_start_offset() {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: weights_file_name.to_owned(),
                tensor_name: (*tensor_name).clone(),
                data_start_offset: tensor_view.data_start_offset(),
                data_end_offset: tensor_view.data_end_offset(),
            });
        }

        let element_count = tensor_view
            .shape
            .iter()
            .try_fold(1_u64, |product, dimension| {
                let dimension = u64::try_from(*dimension)
                    .map_err(|_| ArtifactValidationError::TensorPayloadSizeOverflow)?;
                product
                    .checked_mul(dimension)
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)
            })?;
        let bits_per_element =
            dtype_bits_per_element(&tensor_view.dtype, weights_file_name, tensor_name)?;
        let expected_data_bytes = element_count
            .checked_mul(bits_per_element / 8)
            .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
        let actual_data_bytes = tensor_view
            .data_end_offset()
            .checked_sub(tensor_view.data_start_offset())
            .ok_or_else(|| ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: weights_file_name.to_owned(),
                tensor_name: (*tensor_name).clone(),
                data_start_offset: tensor_view.data_start_offset(),
                data_end_offset: tensor_view.data_end_offset(),
            })?;
        if actual_data_bytes != expected_data_bytes {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: weights_file_name.to_owned(),
                tensor_name: (*tensor_name).clone(),
                data_start_offset: tensor_view.data_start_offset(),
                data_end_offset: tensor_view.data_end_offset(),
            });
        }

        expected_start_offset = tensor_view.data_end_offset();
    }
    Ok(())
}

fn validate_bounded_safetensors_internal_permissive(
    weights_file: &File,
    file_size_bytes: u64,
    weights_file_name: &str,
    tensor_profiles: &[TensorProfile],
) -> Result<BoundedSafetensorsMetadata, ArtifactValidationError> {
    let artifact_safetensors_header =
        read_artifact_safetensors_header(weights_file, file_size_bytes, weights_file_name)?;
    let bounded_metadata = validate_tensors_permissive(
        &artifact_safetensors_header.tensors,
        tensor_profiles,
        artifact_safetensors_header.data_section_start_bytes,
        file_size_bytes,
        weights_file_name,
    )?;
    let actual_payload_bytes = file_size_bytes
        .checked_sub(artifact_safetensors_header.data_section_start_bytes)
        .ok_or(ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: weights_file_name.to_owned(),
            expected_minimum_bytes: artifact_safetensors_header.data_section_start_bytes,
            actual_file_size_bytes: file_size_bytes,
        })?;
    if bounded_metadata.total_payload_bytes != actual_payload_bytes {
        return Err(ArtifactValidationError::SafetensorsPayloadLengthMismatch {
            file_name: weights_file_name.to_owned(),
            declared_payload_bytes: bounded_metadata.total_payload_bytes,
            actual_payload_bytes,
        });
    }
    Ok(bounded_metadata)
}

fn read_artifact_safetensors_header(
    weights_file: &File,
    file_size_bytes: u64,
    weights_file_name: &str,
) -> Result<ArtifactSafetensorsHeader, ArtifactValidationError> {
    let bounded_safetensors_json_header = read_bounded_safetensors_json_header(
        weights_file,
        file_size_bytes,
        MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES,
    )
    .map_err(|bounded_safetensors_header_error| {
        artifact_safetensors_header_error(bounded_safetensors_header_error, weights_file_name)
    })?;
    if let Some(metadata_json_value) = bounded_safetensors_json_header.metadata_json_value {
        serde_json::from_value::<HashMap<String, String>>(metadata_json_value).map_err(
            |source| ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            },
        )?;
    }
    let mut tensors =
        HashMap::with_capacity(bounded_safetensors_json_header.tensor_json_values.len());
    for (tensor_name, tensor_json_value) in bounded_safetensors_json_header.tensor_json_values {
        let tensor_view = serde_json::from_value(tensor_json_value).map_err(|source| {
            ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            }
        })?;
        tensors.insert(tensor_name, tensor_view);
    }
    Ok(ArtifactSafetensorsHeader {
        tensors,
        data_section_start_bytes: bounded_safetensors_json_header.data_section_start_bytes,
    })
}

fn artifact_safetensors_header_error(
    bounded_safetensors_header_error: BoundedSafetensorsHeaderError,
    weights_file_name: &str,
) -> ArtifactValidationError {
    match bounded_safetensors_header_error {
        BoundedSafetensorsHeaderError::ReadLengthPrefix(source) => {
            ArtifactValidationError::ReadSafetensorsLengthPrefix {
                file_name: weights_file_name.to_owned(),
                source,
            }
        }
        BoundedSafetensorsHeaderError::HeaderLengthTooLarge {
            header_length_bytes,
            maximum_header_length_bytes,
        } => ArtifactValidationError::SafetensorsHeaderLengthTooLarge {
            file_name: weights_file_name.to_owned(),
            header_length_bytes,
            maximum_header_length_bytes,
        },
        BoundedSafetensorsHeaderError::HeaderBeyondFile {
            header_end_offset_bytes,
            file_size_bytes,
        } => ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: weights_file_name.to_owned(),
            expected_minimum_bytes: header_end_offset_bytes,
            actual_file_size_bytes: file_size_bytes,
        },
        BoundedSafetensorsHeaderError::ReadHeader(source) => {
            ArtifactValidationError::ReadSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            }
        }
        BoundedSafetensorsHeaderError::InvalidHeaderJson(source) => {
            ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            }
        }
    }
}

/// Validates tensors in a permissive mode: all tensors in the shard header are
/// accepted (no "unexpected tensor" check), and only profiled tensors that are
/// actually present in the shard are validated. This is used for models with
/// embedded vision tensors where the language shard also contains vision tensors
/// that are not in the language tensor profiles.
fn validate_tensors_permissive(
    safetensors_tensors: &HashMap<String, SafetensorsTensorView>,
    tensor_profiles: &[TensorProfile],
    data_section_start: u64,
    file_size_bytes: u64,
    weights_file_name: &str,
) -> Result<BoundedSafetensorsMetadata, ArtifactValidationError> {
    validate_tensor_data_consistency(safetensors_tensors, weights_file_name)?;

    // Validate offsets for ALL tensors in the header, but only dtype/shape
    // for tensors that have profiles.
    for (tensor_name, tensor_view) in safetensors_tensors {
        validate_tensor_offsets(
            tensor_name,
            tensor_view,
            data_section_start,
            file_size_bytes,
            weights_file_name,
        )?;
    }

    // Validate dtype and shape only for profiled tensors present in this shard.
    for tensor_profile in tensor_profiles {
        let Some(tensor_view) = safetensors_tensors.get(&tensor_profile.name) else {
            // This profiled tensor is not in this shard — it's in another shard.
            // This is expected for multi-shard models.
            continue;
        };
        validate_tensor_dtype(tensor_profile, tensor_view, weights_file_name)?;
        validate_tensor_shape(tensor_profile, tensor_view)?;
    }

    let total_payload_bytes =
        safetensors_tensors
            .values()
            .try_fold(0_u64, |total_payload_bytes, tensor_view| {
                let tensor_data_bytes = tensor_view
                    .data_end_offset()
                    .checked_sub(tensor_view.data_start_offset())
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
                total_payload_bytes
                    .checked_add(tensor_data_bytes)
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)
            })?;

    Ok(BoundedSafetensorsMetadata {
        total_payload_bytes,
    })
}

fn validate_all_tensors(
    safetensors_tensors: &HashMap<String, SafetensorsTensorView>,
    expected_tensor_names: &HashSet<&str>,
    tensor_profiles: &[TensorProfile],
    data_section_start: u64,
    file_size_bytes: u64,
    weights_file_name: &str,
) -> Result<BoundedSafetensorsMetadata, ArtifactValidationError> {
    validate_tensor_data_consistency(safetensors_tensors, weights_file_name)?;

    for (tensor_name, tensor_view) in safetensors_tensors {
        if !expected_tensor_names.contains(tensor_name.as_str()) {
            return Err(ArtifactValidationError::UnexpectedTensor {
                tensor_name: tensor_name.clone(),
            });
        }
        validate_tensor_offsets(
            tensor_name,
            tensor_view,
            data_section_start,
            file_size_bytes,
            weights_file_name,
        )?;
    }

    if let Some(missing_tensor_name) = expected_tensor_names
        .iter()
        .find(|tensor_name| !safetensors_tensors.contains_key(**tensor_name))
    {
        return Err(ArtifactValidationError::TensorMissing {
            tensor_name: (*missing_tensor_name).to_owned(),
            file_name: weights_file_name.to_owned(),
        });
    }

    for tensor_profile in tensor_profiles {
        let tensor_view = safetensors_tensors
            .get(&tensor_profile.name)
            .ok_or_else(|| ArtifactValidationError::TensorMissing {
                tensor_name: tensor_profile.name.clone(),
                file_name: weights_file_name.to_owned(),
            })?;
        validate_tensor_dtype(tensor_profile, tensor_view, weights_file_name)?;
        validate_tensor_shape(tensor_profile, tensor_view)?;
    }

    let total_payload_bytes =
        safetensors_tensors
            .values()
            .try_fold(0_u64, |total_payload_bytes, tensor_view| {
                let tensor_data_bytes = tensor_view
                    .data_end_offset()
                    .checked_sub(tensor_view.data_start_offset())
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
                total_payload_bytes
                    .checked_add(tensor_data_bytes)
                    .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)
            })?;

    Ok(BoundedSafetensorsMetadata {
        total_payload_bytes,
    })
}

fn validate_tensor_offsets(
    tensor_name: &str,
    tensor_view: &SafetensorsTensorView,
    data_section_start: u64,
    file_size_bytes: u64,
    file_name: &str,
) -> Result<(), ArtifactValidationError> {
    if tensor_view.data_start_offset() > tensor_view.data_end_offset() {
        return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
            file_name: file_name.to_owned(),
            tensor_name: tensor_name.to_owned(),
            data_start_offset: tensor_view.data_start_offset(),
            data_end_offset: tensor_view.data_end_offset(),
        });
    }
    let absolute_end_offset = data_section_start
        .checked_add(tensor_view.data_end_offset())
        .ok_or_else(|| ArtifactValidationError::SafetensorsOffsetBeyondFile {
            file_name: file_name.to_owned(),
            tensor_name: tensor_name.to_owned(),
            data_end_offset: tensor_view.data_end_offset(),
            file_size_bytes,
        })?;
    if absolute_end_offset > file_size_bytes {
        return Err(ArtifactValidationError::SafetensorsOffsetBeyondFile {
            file_name: file_name.to_owned(),
            tensor_name: tensor_name.to_owned(),
            data_end_offset: tensor_view.data_end_offset(),
            file_size_bytes,
        });
    }
    Ok(())
}

fn validate_tensor_dtype(
    tensor_profile: &TensorProfile,
    tensor_view: &SafetensorsTensorView,
    file_name: &str,
) -> Result<(), ArtifactValidationError> {
    let tensor_dtype_matches_profile = match tensor_profile.dtype {
        TensorDtype::BFloat16 => tensor_view.dtype == "BF16",
        TensorDtype::BFloat16OrFloat32 => matches!(tensor_view.dtype.as_str(), "BF16" | "F32"),
        TensorDtype::Float32 => tensor_view.dtype == "F32",
        TensorDtype::UInt32 => tensor_view.dtype == "U32",
    };
    if !tensor_dtype_matches_profile {
        let actual_dtype =
            parse_safetensors_dtype(&tensor_view.dtype, file_name, &tensor_profile.name)?;
        return Err(ArtifactValidationError::TensorDtypeMismatch {
            tensor_name: tensor_profile.name.clone(),
            expected_dtype: tensor_profile.dtype,
            actual_dtype,
        });
    }
    Ok(())
}

fn validate_tensor_shape(
    tensor_profile: &TensorProfile,
    tensor_view: &SafetensorsTensorView,
) -> Result<(), ArtifactValidationError> {
    if tensor_view.shape != tensor_profile.shape {
        return Err(ArtifactValidationError::TensorShapeMismatch {
            tensor_name: tensor_profile.name.clone(),
            expected_shape: tensor_profile.shape.clone(),
            actual_shape: tensor_view.shape.clone(),
        });
    }
    Ok(())
}
