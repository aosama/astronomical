//! Descriptor construction for one expert's slice inside a per-expert pack.

use std::{fs, time::UNIX_EPOCH};

use astronomical_model_serving::QuantizedTensorSource;

use crate::aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackError,
    AlignedExpertPackTensorDescriptor,
};
use crate::aligned_expert_pack_layout::validate_segment_extent;
use serde::{Deserialize, Serialize};

/// One tensor-major segment holding a single expert's payload.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PerExpertPackTensorDescriptor {
    pub tensor_name: String,
    pub projection_name: String,
    pub parameter_name: String,
    pub dtype_name: String,
    pub expert_local_shape: Vec<usize>,
    pub source_file_name: String,
    pub source_file_size_bytes: u64,
    pub source_file_modified_unix_nanoseconds: u64,
    pub source_payload_offset_bytes: u64,
    pub source_expert_payload_offset_bytes: u64,
    pub bytes_per_expert: usize,
    pub pack_segment_offset_bytes: u64,
    pub logical_byte_count: usize,
    pub padded_segment_byte_count: u64,
}

pub(super) fn descriptor_from_source_for_expert(
    tensor_source: &QuantizedTensorSource,
    expert_id: usize,
    pack_segment_offset_bytes: u64,
    expected_expert_capacity: usize,
) -> Result<PerExpertPackTensorDescriptor, AlignedExpertPackError> {
    if tensor_source.expert_capacity != expected_expert_capacity
        || tensor_source.full_shape.first().copied() != Some(expected_expert_capacity)
    {
        return Err(AlignedExpertPackError::InvalidTensorShapeByteCount {
            tensor_name: tensor_source.tensor_name.clone(),
        });
    }
    let bytes_per_expert = tensor_source.bytes_per_expert;
    let source_expert_payload_offset_bytes = tensor_source
        .tensor_payload_offset
        .checked_add(
            u64::try_from(expert_id)
                .ok()
                .and_then(|selected_expert_id| {
                    selected_expert_id.checked_mul(bytes_per_expert as u64)
                })
                .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                    operation: "calculate a per-expert source payload offset",
                })?,
        )
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate a per-expert source payload offset",
        })?;
    let padded_segment_byte_count = align_up(
        u64::try_from(bytes_per_expert).map_err(|_| {
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert a per-expert tensor byte count",
            }
        })?,
        ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES,
    )?;
    let mut expert_local_shape = tensor_source.full_shape.clone();
    if !expert_local_shape.is_empty() {
        expert_local_shape.remove(0);
    }
    let source_file_name = tensor_source
        .source_file
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| {
            AlignedExpertPackError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "per-expert source file must have a UTF-8 file name",
            ))
        })?
        .to_owned();
    Ok(PerExpertPackTensorDescriptor {
        tensor_name: tensor_source.tensor_name.clone(),
        projection_name: tensor_source.projection_name.clone(),
        parameter_name: tensor_source.parameter_name.clone(),
        dtype_name: tensor_source.dtype.as_str().to_owned(),
        expert_local_shape,
        source_file_name,
        source_file_size_bytes: tensor_source.source_file_size_bytes,
        source_file_modified_unix_nanoseconds: source_file_modified_unix_nanoseconds(
            tensor_source,
        )?,
        source_payload_offset_bytes: tensor_source.tensor_payload_offset,
        source_expert_payload_offset_bytes,
        bytes_per_expert,
        pack_segment_offset_bytes,
        logical_byte_count: bytes_per_expert,
        padded_segment_byte_count,
    })
}

pub(super) fn validate_source_file_range_for_expert(
    tensor_source: &QuantizedTensorSource,
    tensor_descriptor: &PerExpertPackTensorDescriptor,
) -> Result<(), AlignedExpertPackError> {
    let actual_source_file_size_bytes = fs::metadata(&tensor_source.source_file)?.len();
    let source_end_offset_bytes = tensor_descriptor
        .source_expert_payload_offset_bytes
        .checked_add(
            u64::try_from(tensor_descriptor.logical_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert a per-expert source tensor byte count",
                }
            })?,
        )
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate a per-expert source tensor end offset",
        })?;
    if actual_source_file_size_bytes != tensor_descriptor.source_file_size_bytes
        || source_end_offset_bytes > actual_source_file_size_bytes
    {
        return Err(AlignedExpertPackError::SourceRangeExceedsFile {
            tensor_name: tensor_descriptor.tensor_name.clone(),
        });
    }
    validate_segment_extent(
        &AlignedExpertPackTensorDescriptor {
            tensor_name: tensor_descriptor.tensor_name.clone(),
            projection_name: tensor_descriptor.projection_name.clone(),
            parameter_name: tensor_descriptor.parameter_name.clone(),
            dtype_name: tensor_descriptor.dtype_name.clone(),
            full_shape: tensor_descriptor.expert_local_shape.clone(),
            source_file_name: tensor_descriptor.source_file_name.clone(),
            source_file_size_bytes: tensor_descriptor.source_file_size_bytes,
            source_file_modified_unix_nanoseconds: tensor_descriptor
                .source_file_modified_unix_nanoseconds,
            source_payload_offset_bytes: tensor_descriptor.source_expert_payload_offset_bytes,
            bytes_per_expert: tensor_descriptor.bytes_per_expert,
            pack_segment_offset_bytes: tensor_descriptor.pack_segment_offset_bytes,
            logical_byte_count: tensor_descriptor.logical_byte_count,
            padded_segment_byte_count: tensor_descriptor.padded_segment_byte_count,
        },
        tensor_descriptor.pack_segment_offset_bytes,
    )
}

fn source_file_modified_unix_nanoseconds(
    tensor_source: &QuantizedTensorSource,
) -> Result<u64, AlignedExpertPackError> {
    let modified_duration = fs::metadata(&tensor_source.source_file)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map_err(|duration_error| {
            AlignedExpertPackError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "per-expert source modification time precedes Unix epoch: {duration_error}"
                ),
            ))
        })?;
    u64::try_from(modified_duration.as_nanos()).map_err(|_| {
        AlignedExpertPackError::ArithmeticOverflow {
            operation: "convert per-expert source modification nanoseconds",
        }
    })
}

pub(super) fn align_up(
    byte_count: u64,
    alignment_bytes: u64,
) -> Result<u64, AlignedExpertPackError> {
    let alignment_remainder = byte_count % alignment_bytes;
    if alignment_remainder == 0 {
        return Ok(byte_count);
    }
    byte_count
        .checked_add(alignment_bytes - alignment_remainder)
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "align a per-expert pack byte count",
        })
}
