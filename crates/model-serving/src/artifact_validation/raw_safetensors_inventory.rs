//! Strict family-neutral inventory of one retained SafeTensors shard.

use std::collections::HashMap;

use ::safetensors::Dtype;

use super::bounded_safetensors::{
    MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES, artifact_safetensors_header_error,
};
use super::safetensors_dtype::{checked_safetensors_payload_bytes, parse_raw_safetensors_dtype};
use super::{ArtifactValidationError, ValidatedRequiredFile};
use crate::safetensors::{SafetensorsTensorView, read_bounded_safetensors_json_header};

/// A deterministic raw inventory produced before family normalization.
#[derive(Debug)]
pub(crate) struct RawSafetensorsInventory {
    /// Tensor descriptors sorted lexically by their unmodified raw names.
    pub(crate) tensor_descriptors: Vec<RawSafetensorsTensorDescriptor>,
    /// Exact bytes covered by all contiguous tensor payload intervals.
    pub(crate) shard_payload_bytes: u64,
}

/// One format-valid tensor declaration with absolute source-file intervals.
#[derive(Clone, Debug)]
pub(crate) struct RawSafetensorsTensorDescriptor {
    /// Unmodified tensor key from the SafeTensors header.
    pub(crate) tensor_name: String,
    /// Actual storage dtype declared by the shard.
    pub(crate) dtype: Dtype,
    /// Actual row-major tensor dimensions declared by the shard.
    pub(crate) shape: Vec<usize>,
    /// Inclusive byte offset from the beginning of the retained file.
    pub(crate) data_start_offset_bytes: u64,
    /// Exclusive byte offset from the beginning of the retained file.
    pub(crate) data_end_offset_bytes: u64,
    /// Exact checked bytes in this tensor's source interval.
    pub(crate) tensor_payload_bytes: u64,
}

/// Parsed declaration retained until aggregate accounting is complete.
struct RawSafetensorsTensorDeclaration {
    tensor_name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data_start_offset: u64,
    data_end_offset: u64,
    tensor_payload_bytes: u64,
}

/// Reads and validates raw tensor declarations from one retained artifact descriptor.
pub(crate) fn read_raw_safetensors_inventory(
    validated_required_file: &ValidatedRequiredFile,
) -> Result<RawSafetensorsInventory, ArtifactValidationError> {
    let weights_file_name = validated_required_file.file_name();
    let file_size_bytes = validated_required_file.size_bytes();
    let bounded_header = read_bounded_safetensors_json_header(
        validated_required_file.file(),
        file_size_bytes,
        MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES,
    )
    .map_err(|header_error| artifact_safetensors_header_error(header_error, weights_file_name))?;

    // Metadata remains validated as a string map but never enters the tensor
    // inventory consumed by family-owned normalization.
    if let Some(metadata_json_value) = bounded_header.metadata_json_value {
        serde_json::from_value::<HashMap<String, String>>(metadata_json_value).map_err(
            |source| ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            },
        )?;
    }

    let data_section_start_bytes = bounded_header.data_section_start_bytes;
    let tensor_count = bounded_header.tensor_json_values.len();
    let mut tensor_declarations = Vec::with_capacity(tensor_count);
    let mut shard_payload_bytes = 0_u64;
    for (tensor_name, tensor_json_value) in bounded_header.tensor_json_values {
        validate_tensor_name(&tensor_name, weights_file_name)?;
        let tensor_view: SafetensorsTensorView = serde_json::from_value(tensor_json_value)
            .map_err(|source| ArtifactValidationError::InvalidSafetensorsHeader {
                file_name: weights_file_name.to_owned(),
                source,
            })?;
        let dtype =
            parse_raw_safetensors_dtype(&tensor_view.dtype, weights_file_name, &tensor_name)?;
        let tensor_payload_bytes =
            validate_tensor_payload_bytes(&tensor_name, &tensor_view, dtype, weights_file_name)?;
        shard_payload_bytes = shard_payload_bytes
            .checked_add(tensor_payload_bytes)
            .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)?;
        let data_start_offset = tensor_view.data_start_offset();
        let data_end_offset = tensor_view.data_end_offset();
        tensor_declarations.push(RawSafetensorsTensorDeclaration {
            tensor_name,
            dtype,
            shape: tensor_view.shape,
            data_start_offset,
            data_end_offset,
            tensor_payload_bytes,
        });
    }

    let mut tensor_descriptors = Vec::with_capacity(tensor_count);
    for tensor_declaration in tensor_declarations {
        let data_start_offset_bytes = absolute_tensor_offset(
            data_section_start_bytes,
            tensor_declaration.data_start_offset,
            tensor_declaration.data_end_offset,
            file_size_bytes,
            weights_file_name,
            &tensor_declaration.tensor_name,
        )?;
        let data_end_offset_bytes = absolute_tensor_offset(
            data_section_start_bytes,
            tensor_declaration.data_end_offset,
            tensor_declaration.data_end_offset,
            file_size_bytes,
            weights_file_name,
            &tensor_declaration.tensor_name,
        )?;
        tensor_descriptors.push(RawSafetensorsTensorDescriptor {
            tensor_name: tensor_declaration.tensor_name,
            dtype: tensor_declaration.dtype,
            shape: tensor_declaration.shape,
            data_start_offset_bytes,
            data_end_offset_bytes,
            tensor_payload_bytes: tensor_declaration.tensor_payload_bytes,
        });
    }

    validate_contiguous_intervals(
        &tensor_descriptors,
        data_section_start_bytes,
        weights_file_name,
    )?;
    let actual_payload_bytes = file_size_bytes
        .checked_sub(data_section_start_bytes)
        .ok_or(ArtifactValidationError::TruncatedSafetensorsFile {
            file_name: weights_file_name.to_owned(),
            expected_minimum_bytes: data_section_start_bytes,
            actual_file_size_bytes: file_size_bytes,
        })?;
    if shard_payload_bytes != actual_payload_bytes {
        return Err(ArtifactValidationError::SafetensorsPayloadLengthMismatch {
            file_name: weights_file_name.to_owned(),
            declared_payload_bytes: shard_payload_bytes,
            actual_payload_bytes,
        });
    }

    tensor_descriptors.sort_by(|left_descriptor, right_descriptor| {
        left_descriptor
            .tensor_name
            .cmp(&right_descriptor.tensor_name)
    });
    Ok(RawSafetensorsInventory {
        tensor_descriptors,
        shard_payload_bytes,
    })
}

fn validate_tensor_name(
    tensor_name: &str,
    weights_file_name: &str,
) -> Result<(), ArtifactValidationError> {
    let tensor_name_length_bytes = u64::try_from(tensor_name.len())
        .map_err(|_| ArtifactValidationError::TensorPayloadSizeOverflow)?;
    if tensor_name_length_bytes == 0
        || tensor_name_length_bytes > MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES
    {
        return Err(ArtifactValidationError::InvalidSafetensorsTensorName {
            file_name: weights_file_name.to_owned(),
            tensor_name_length_bytes,
            maximum_tensor_name_length_bytes: MAXIMUM_ARTIFACT_SAFETENSORS_HEADER_LENGTH_BYTES,
        });
    }
    Ok(())
}

fn validate_tensor_payload_bytes(
    tensor_name: &str,
    tensor_view: &SafetensorsTensorView,
    dtype: Dtype,
    weights_file_name: &str,
) -> Result<u64, ArtifactValidationError> {
    let element_count = tensor_view
        .shape
        .iter()
        .try_fold(1_u64, |shape_product, dimension| {
            let dimension = u64::try_from(*dimension)
                .map_err(|_| ArtifactValidationError::TensorPayloadSizeOverflow)?;
            shape_product
                .checked_mul(dimension)
                .ok_or(ArtifactValidationError::TensorPayloadSizeOverflow)
        })?;
    let dtype_bits = u64::try_from(dtype.bitsize())
        .map_err(|_| ArtifactValidationError::TensorPayloadSizeOverflow)?;
    let expected_payload_bytes = checked_safetensors_payload_bytes(element_count, dtype_bits)?;
    let actual_payload_bytes = tensor_view
        .data_end_offset()
        .checked_sub(tensor_view.data_start_offset())
        .ok_or_else(|| invalid_data_offsets(weights_file_name, tensor_name, tensor_view))?;
    if expected_payload_bytes != Some(actual_payload_bytes) {
        return Err(invalid_data_offsets(
            weights_file_name,
            tensor_name,
            tensor_view,
        ));
    }
    Ok(actual_payload_bytes)
}

fn absolute_tensor_offset(
    data_section_start_bytes: u64,
    relative_offset_bytes: u64,
    data_end_offset: u64,
    file_size_bytes: u64,
    weights_file_name: &str,
    tensor_name: &str,
) -> Result<u64, ArtifactValidationError> {
    let absolute_offset_bytes = data_section_start_bytes
        .checked_add(relative_offset_bytes)
        .ok_or_else(|| {
            offset_beyond_file(
                weights_file_name,
                tensor_name,
                data_end_offset,
                file_size_bytes,
            )
        })?;
    if absolute_offset_bytes > file_size_bytes {
        return Err(offset_beyond_file(
            weights_file_name,
            tensor_name,
            data_end_offset,
            file_size_bytes,
        ));
    }
    Ok(absolute_offset_bytes)
}

fn validate_contiguous_intervals(
    tensor_descriptors: &[RawSafetensorsTensorDescriptor],
    data_section_start_bytes: u64,
    weights_file_name: &str,
) -> Result<(), ArtifactValidationError> {
    let mut interval_order = tensor_descriptors.iter().collect::<Vec<_>>();
    interval_order.sort_by(|left_descriptor, right_descriptor| {
        left_descriptor
            .data_start_offset_bytes
            .cmp(&right_descriptor.data_start_offset_bytes)
            .then_with(|| {
                left_descriptor
                    .tensor_name
                    .cmp(&right_descriptor.tensor_name)
            })
    });
    let mut expected_start_offset_bytes = data_section_start_bytes;
    for tensor_descriptor in interval_order {
        if tensor_descriptor.data_start_offset_bytes != expected_start_offset_bytes {
            return Err(ArtifactValidationError::SafetensorsInvalidDataOffsets {
                file_name: weights_file_name.to_owned(),
                tensor_name: tensor_descriptor.tensor_name.clone(),
                data_start_offset: tensor_descriptor
                    .data_start_offset_bytes
                    .saturating_sub(data_section_start_bytes),
                data_end_offset: tensor_descriptor
                    .data_end_offset_bytes
                    .saturating_sub(data_section_start_bytes),
            });
        }
        expected_start_offset_bytes = tensor_descriptor.data_end_offset_bytes;
    }
    Ok(())
}

fn invalid_data_offsets(
    weights_file_name: &str,
    tensor_name: &str,
    tensor_view: &SafetensorsTensorView,
) -> ArtifactValidationError {
    ArtifactValidationError::SafetensorsInvalidDataOffsets {
        file_name: weights_file_name.to_owned(),
        tensor_name: tensor_name.to_owned(),
        data_start_offset: tensor_view.data_start_offset(),
        data_end_offset: tensor_view.data_end_offset(),
    }
}

fn offset_beyond_file(
    weights_file_name: &str,
    tensor_name: &str,
    data_end_offset: u64,
    file_size_bytes: u64,
) -> ArtifactValidationError {
    ArtifactValidationError::SafetensorsOffsetBeyondFile {
        file_name: weights_file_name.to_owned(),
        tensor_name: tensor_name.to_owned(),
        data_end_offset,
        file_size_bytes,
    }
}
