use std::{fs, time::UNIX_EPOCH};

use astronomical_model_serving::{QuantizedExpertLayerPlan, QuantizedTensorSource};

use crate::aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES, AlignedExpertPackError,
    AlignedExpertPackTensorDescriptor,
};

pub(super) fn descriptor_from_source(
    tensor_source: &QuantizedTensorSource,
    pack_segment_offset_bytes: u64,
    expected_expert_capacity: usize,
) -> Result<AlignedExpertPackTensorDescriptor, AlignedExpertPackError> {
    let shape_element_count = tensor_source
        .full_shape
        .iter()
        .try_fold(1_usize, |shape_product, dimension| {
            shape_product.checked_mul(*dimension)
        })
        .ok_or(AlignedExpertPackError::InvalidTensorShapeByteCount {
            tensor_name: tensor_source.tensor_name.clone(),
        })?;
    let logical_byte_count = shape_element_count
        .checked_mul(tensor_source.dtype.byte_width())
        .ok_or(AlignedExpertPackError::InvalidTensorShapeByteCount {
            tensor_name: tensor_source.tensor_name.clone(),
        })?;
    let expected_logical_byte_count = expected_expert_capacity
        .checked_mul(tensor_source.bytes_per_expert)
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate an expected aligned expert tensor byte count",
        })?;
    if tensor_source.expert_capacity != expected_expert_capacity
        || logical_byte_count != expected_logical_byte_count
    {
        return Err(AlignedExpertPackError::InvalidTensorShapeByteCount {
            tensor_name: tensor_source.tensor_name.clone(),
        });
    }
    let padded_segment_byte_count = align_up(
        u64::try_from(logical_byte_count).map_err(|_| {
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert an aligned expert tensor byte count",
            }
        })?,
        ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES,
    )?;
    Ok(AlignedExpertPackTensorDescriptor {
        tensor_name: tensor_source.tensor_name.clone(),
        projection_name: tensor_source.projection_name.clone(),
        parameter_name: tensor_source.parameter_name.clone(),
        dtype_name: tensor_source.dtype.as_str().to_owned(),
        full_shape: tensor_source.full_shape.clone(),
        source_file_name: tensor_source
            .source_file
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .ok_or_else(|| {
                AlignedExpertPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "aligned expert source file must have a UTF-8 file name",
                ))
            })?
            .to_owned(),
        source_file_size_bytes: tensor_source.source_file_size_bytes,
        source_file_modified_unix_nanoseconds: source_file_modified_unix_nanoseconds(
            tensor_source,
        )?,
        source_payload_offset_bytes: tensor_source.tensor_payload_offset,
        bytes_per_expert: tensor_source.bytes_per_expert,
        pack_segment_offset_bytes,
        logical_byte_count,
        padded_segment_byte_count,
    })
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
                    "aligned expert source modification time precedes Unix epoch: {duration_error}"
                ),
            ))
        })?;
    u64::try_from(modified_duration.as_nanos()).map_err(|_| {
        AlignedExpertPackError::ArithmeticOverflow {
            operation: "convert aligned expert source modification nanoseconds",
        }
    })
}

pub(super) fn ordered_tensor_sources(
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<Vec<&QuantizedTensorSource>, AlignedExpertPackError> {
    let mut ordered_tensor_sources = layer_plan.tensor_sources.iter().collect::<Vec<_>>();
    ordered_tensor_sources.sort_by_key(|tensor_source| {
        (
            projection_sort_position(&tensor_source.projection_name),
            parameter_sort_position(&tensor_source.parameter_name),
        )
    });
    for adjacent_tensor_sources in ordered_tensor_sources.windows(2) {
        if adjacent_tensor_sources[0].projection_name == adjacent_tensor_sources[1].projection_name
            && adjacent_tensor_sources[0].parameter_name
                == adjacent_tensor_sources[1].parameter_name
        {
            return Err(AlignedExpertPackError::ForeignTensorDescriptor {
                tensor_name: adjacent_tensor_sources[0].tensor_name.clone(),
            });
        }
    }
    Ok(ordered_tensor_sources)
}

fn projection_sort_position(projection_name: &str) -> u8 {
    match projection_name {
        "gate_proj" => 0,
        "up_proj" => 1,
        "down_proj" => 2,
        _ => 3,
    }
}

fn parameter_sort_position(parameter_name: &str) -> u8 {
    match parameter_name {
        "weight" => 0,
        "scales" => 1,
        "biases" => 2,
        _ => 3,
    }
}

pub(super) fn validate_source_file_range(
    tensor_source: &QuantizedTensorSource,
    tensor_descriptor: &AlignedExpertPackTensorDescriptor,
) -> Result<(), AlignedExpertPackError> {
    let actual_source_file_size_bytes = fs::metadata(&tensor_source.source_file)?.len();
    let source_end_offset_bytes = tensor_descriptor
        .source_payload_offset_bytes
        .checked_add(
            u64::try_from(tensor_descriptor.logical_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert an aligned expert source tensor byte count",
                }
            })?,
        )
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate an aligned expert source tensor end offset",
        })?;
    if actual_source_file_size_bytes != tensor_descriptor.source_file_size_bytes
        || source_end_offset_bytes > actual_source_file_size_bytes
    {
        return Err(AlignedExpertPackError::SourceRangeExceedsFile {
            tensor_name: tensor_descriptor.tensor_name.clone(),
        });
    }
    Ok(())
}

pub(super) fn validate_segment_extent(
    tensor_descriptor: &AlignedExpertPackTensorDescriptor,
    expected_segment_offset_bytes: u64,
) -> Result<(), AlignedExpertPackError> {
    if tensor_descriptor.pack_segment_offset_bytes < expected_segment_offset_bytes {
        return Err(AlignedExpertPackError::OverlappingSegment {
            tensor_name: tensor_descriptor.tensor_name.clone(),
        });
    }
    if !tensor_descriptor
        .pack_segment_offset_bytes
        .is_multiple_of(ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES)
        || tensor_descriptor.pack_segment_offset_bytes != expected_segment_offset_bytes
    {
        return Err(AlignedExpertPackError::UnalignedSegmentOffset {
            tensor_name: tensor_descriptor.tensor_name.clone(),
        });
    }
    let expected_padded_segment_byte_count = align_up(
        u64::try_from(tensor_descriptor.logical_byte_count).map_err(|_| {
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert a packed expert tensor byte count",
            }
        })?,
        ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES,
    )?;
    if tensor_descriptor.padded_segment_byte_count != expected_padded_segment_byte_count {
        return Err(AlignedExpertPackError::InvalidPaddedSegmentExtent {
            tensor_name: tensor_descriptor.tensor_name.clone(),
        });
    }
    Ok(())
}

fn align_up(byte_count: u64, alignment_bytes: u64) -> Result<u64, AlignedExpertPackError> {
    let alignment_remainder = byte_count % alignment_bytes;
    if alignment_remainder == 0 {
        return Ok(byte_count);
    }
    byte_count
        .checked_add(alignment_bytes - alignment_remainder)
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "align an expert pack byte count",
        })
}
