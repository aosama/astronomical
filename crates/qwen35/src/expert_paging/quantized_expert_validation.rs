//! Pure validation functions for quantized expert manifest construction.
//!
//! Extracted from `quantized_expert_manifest` to keep file sizes under the
//! 500-line coding principle. These functions have no I/O dependencies and
//! operate solely on their arguments.

use super::quantized_expert_manifest::{ExpertManifestError, QuantizationMode};

/// Validates that expert IDs are non-empty, strictly ascending, and within capacity.
pub fn validate_expert_ids(
    expert_ids: &[usize],
    expert_capacity: usize,
) -> Result<Vec<usize>, ExpertManifestError> {
    if expert_ids.is_empty() {
        return Err(ExpertManifestError::EmptyExpertIds);
    }
    if expert_ids
        .iter()
        .any(|expert_id| *expert_id >= expert_capacity)
    {
        let Some(maximum_selected_expert_id) = expert_ids.iter().copied().max() else {
            return Err(ExpertManifestError::EmptyExpertIds);
        };
        return Err(ExpertManifestError::ExpertIdExceedsCapacity {
            max_selected_id: maximum_selected_expert_id,
            expert_capacity,
        });
    }
    for adjacent_expert_ids in expert_ids.windows(2) {
        if adjacent_expert_ids[0] >= adjacent_expert_ids[1] {
            return Err(ExpertManifestError::NonAscendingExpertIds);
        }
    }
    Ok(expert_ids.to_vec())
}

/// Validates quantization contract parameters: affine mode, positive bits/group_size,
/// and that groups pack into whole U32 elements.
pub fn validate_quantization_contract(
    bits: i32,
    group_size: i32,
    mode: QuantizationMode,
) -> Result<(), ExpertManifestError> {
    if mode != QuantizationMode::Affine {
        return Err(ExpertManifestError::UnsupportedQuantizationMode {
            mode: format!("{mode:?}"),
        });
    }
    if bits <= 0 {
        return Err(ExpertManifestError::InvalidBits);
    }
    if group_size <= 0 {
        return Err(ExpertManifestError::InvalidGroupSize);
    }
    let Some(packed_group_bit_count) = group_size.checked_mul(bits) else {
        return Err(ExpertManifestError::GroupsNotPackedIntoU32 { bits, group_size });
    };
    if packed_group_bit_count % 32 != 0 {
        return Err(ExpertManifestError::GroupsNotPackedIntoU32 { bits, group_size });
    }
    Ok(())
}

/// Rejects overlapping or zero-length source intervals before native loading.
pub fn validate_source_intervals(
    source_intervals: &[super::quantized_expert_manifest::QuantizedExpertSourceInterval],
    _source_file_size_bytes: u64,
) -> Result<(), ExpertManifestError> {
    let mut previous_source_end = None;
    for source_interval in source_intervals {
        if source_interval.source_byte_count == 0 {
            return Err(ExpertManifestError::SourceIntervalExceedsFile {
                source_file_offset: source_interval.source_file_offset,
                source_byte_count: source_interval.source_byte_count,
                source_file_size_bytes: 0,
            });
        }
        let source_byte_count = u64::try_from(source_interval.source_byte_count).map_err(|_| {
            ExpertManifestError::SourceIntervalExceedsFile {
                source_file_offset: source_interval.source_file_offset,
                source_byte_count: source_interval.source_byte_count,
                source_file_size_bytes: 0,
            }
        })?;
        let source_end = source_interval
            .source_file_offset
            .checked_add(source_byte_count)
            .ok_or(ExpertManifestError::SourceIntervalExceedsFile {
                source_file_offset: source_interval.source_file_offset,
                source_byte_count: source_interval.source_byte_count,
                source_file_size_bytes: 0,
            })?;
        if previous_source_end.is_some_and(|previous_source_end| {
            source_interval.source_file_offset < previous_source_end
        }) {
            return Err(ExpertManifestError::OverlappingSourceIntervals {
                source_file_offset: source_interval.source_file_offset,
            });
        }
        previous_source_end = Some(source_end);
    }
    Ok(())
}

/// Requires exact compact virtual coverage without omitted-byte gaps.
pub fn validate_virtual_intervals(
    source_intervals: &[super::quantized_expert_manifest::QuantizedExpertSourceInterval],
    virtual_payload_byte_count: u64,
) -> Result<(), ExpertManifestError> {
    let mut sorted_source_intervals: Vec<
        &super::quantized_expert_manifest::QuantizedExpertSourceInterval,
    > = source_intervals.iter().collect();
    sorted_source_intervals.sort_by_key(|source_interval| source_interval.virtual_payload_offset);
    let mut expected_virtual_offset: u64 = 0;
    for source_interval in sorted_source_intervals {
        if source_interval.virtual_payload_offset != expected_virtual_offset {
            return Err(ExpertManifestError::NonContiguousVirtualIntervals {
                expected_virtual_offset,
                actual_virtual_offset: source_interval.virtual_payload_offset,
            });
        }
        let source_byte_count = u64::try_from(source_interval.source_byte_count).map_err(|_| {
            ExpertManifestError::VirtualIntervalsShortfall {
                declared_bytes: virtual_payload_byte_count,
                actual_bytes: u64::MAX,
            }
        })?;
        expected_virtual_offset = expected_virtual_offset
            .checked_add(source_byte_count)
            .ok_or(ExpertManifestError::VirtualIntervalsShortfall {
                declared_bytes: virtual_payload_byte_count,
                actual_bytes: u64::MAX,
            })?;
    }
    if expected_virtual_offset != virtual_payload_byte_count {
        return Err(ExpertManifestError::VirtualIntervalsShortfall {
            declared_bytes: virtual_payload_byte_count,
            actual_bytes: expected_virtual_offset,
        });
    }
    Ok(())
}
