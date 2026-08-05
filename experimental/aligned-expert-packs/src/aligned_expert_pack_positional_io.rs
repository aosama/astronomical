use std::{fs::File, os::unix::fs::FileExt};

use astronomical_model_serving::QuantizedTensorSource;

use crate::aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_HEADER_BYTES, ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES,
    ALIGNED_EXPERT_PACK_MAGIC, AlignedExpertPackError, AlignedExpertPackTensorDescriptor,
};

const PACK_COPY_SCRATCH_BYTES: usize = 64 * 1024;

pub(super) fn write_header_region(
    aligned_expert_pack_file: &File,
    serialized_header_payload: &[u8],
) -> Result<(), AlignedExpertPackError> {
    let header_payload_byte_count =
        u64::try_from(serialized_header_payload.len()).map_err(|_| {
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert an aligned expert pack header payload length",
            }
        })?;
    let mut header_region_bytes = vec![0_u8; ALIGNED_EXPERT_PACK_HEADER_BYTES as usize];
    header_region_bytes[..ALIGNED_EXPERT_PACK_MAGIC.len()]
        .copy_from_slice(&ALIGNED_EXPERT_PACK_MAGIC);
    header_region_bytes[ALIGNED_EXPERT_PACK_MAGIC.len()..ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES]
        .copy_from_slice(&header_payload_byte_count.to_le_bytes());
    let header_payload_end_offset = ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES
        .checked_add(serialized_header_payload.len())
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate an aligned expert pack header payload end offset",
        })?;
    if header_payload_end_offset > header_region_bytes.len() {
        return Err(AlignedExpertPackError::HeaderPayloadTooLarge {
            header_payload_byte_count,
        });
    }
    header_region_bytes[ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES..header_payload_end_offset]
        .copy_from_slice(serialized_header_payload);
    write_all_at(aligned_expert_pack_file, &header_region_bytes, 0)
}

pub(super) fn copy_source_tensor_to_pack(
    tensor_source: &QuantizedTensorSource,
    tensor_descriptor: &AlignedExpertPackTensorDescriptor,
    aligned_expert_pack_file: &File,
) -> Result<(), AlignedExpertPackError> {
    let source_file = File::open(&tensor_source.source_file)?;
    let mut source_copy_scratch_bytes = vec![0_u8; PACK_COPY_SCRATCH_BYTES];
    let mut copied_byte_count = 0_usize;
    while copied_byte_count < tensor_descriptor.logical_byte_count {
        let remaining_byte_count = tensor_descriptor.logical_byte_count - copied_byte_count;
        let next_copy_byte_count = remaining_byte_count.min(source_copy_scratch_bytes.len());
        let source_offset_bytes = tensor_descriptor
            .source_payload_offset_bytes
            .checked_add(u64::try_from(copied_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert an aligned expert source copy offset",
                }
            })?)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate an aligned expert source copy offset",
            })?;
        read_exact_at(
            &source_file,
            &mut source_copy_scratch_bytes[..next_copy_byte_count],
            source_offset_bytes,
        )?;
        let destination_offset_bytes = tensor_descriptor
            .pack_segment_offset_bytes
            .checked_add(u64::try_from(copied_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert an aligned expert destination copy offset",
                }
            })?)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate an aligned expert destination copy offset",
            })?;
        write_all_at(
            aligned_expert_pack_file,
            &source_copy_scratch_bytes[..next_copy_byte_count],
            destination_offset_bytes,
        )?;
        copied_byte_count = copied_byte_count.checked_add(next_copy_byte_count).ok_or(
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "advance an aligned expert tensor copy cursor",
            },
        )?;
    }
    Ok(())
}

pub(super) fn compare_source_tensor_to_pack(
    tensor_source: &QuantizedTensorSource,
    tensor_descriptor: &AlignedExpertPackTensorDescriptor,
    aligned_expert_pack_file: &File,
) -> Result<(), AlignedExpertPackError> {
    let source_file = File::open(&tensor_source.source_file)?;
    let mut source_comparison_bytes = vec![0_u8; PACK_COPY_SCRATCH_BYTES];
    let mut pack_comparison_bytes = vec![0_u8; PACK_COPY_SCRATCH_BYTES];
    let mut compared_byte_count = 0_usize;
    while compared_byte_count < tensor_descriptor.logical_byte_count {
        let remaining_byte_count = tensor_descriptor.logical_byte_count - compared_byte_count;
        let next_comparison_byte_count = remaining_byte_count.min(PACK_COPY_SCRATCH_BYTES);
        let compared_byte_count_u64 = u64::try_from(compared_byte_count).map_err(|_| {
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert aligned expert payload comparison offset",
            }
        })?;
        read_exact_at(
            &source_file,
            &mut source_comparison_bytes[..next_comparison_byte_count],
            tensor_descriptor
                .source_payload_offset_bytes
                .checked_add(compared_byte_count_u64)
                .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                    operation: "calculate aligned expert source comparison offset",
                })?,
        )?;
        read_exact_at(
            aligned_expert_pack_file,
            &mut pack_comparison_bytes[..next_comparison_byte_count],
            tensor_descriptor
                .pack_segment_offset_bytes
                .checked_add(compared_byte_count_u64)
                .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                    operation: "calculate aligned expert pack comparison offset",
                })?,
        )?;
        if source_comparison_bytes[..next_comparison_byte_count]
            != pack_comparison_bytes[..next_comparison_byte_count]
        {
            let mismatch_position = source_comparison_bytes[..next_comparison_byte_count]
                .iter()
                .zip(&pack_comparison_bytes[..next_comparison_byte_count])
                .position(|(source_byte, pack_byte)| source_byte != pack_byte)
                .unwrap_or(0);
            return Err(AlignedExpertPackError::PayloadByteMismatch {
                tensor_name: tensor_descriptor.tensor_name.clone(),
                tensor_byte_offset: compared_byte_count_u64
                    .saturating_add(mismatch_position as u64),
            });
        }
        compared_byte_count = compared_byte_count
            .checked_add(next_comparison_byte_count)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "advance aligned expert payload comparison cursor",
            })?;
    }
    Ok(())
}

fn read_exact_at(
    source_file: &File,
    destination_bytes: &mut [u8],
    source_offset_bytes: u64,
) -> Result<(), AlignedExpertPackError> {
    let mut consumed_byte_count = 0_usize;
    while consumed_byte_count < destination_bytes.len() {
        let read_offset_bytes = source_offset_bytes
            .checked_add(u64::try_from(consumed_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert a positional source read offset",
                }
            })?)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate a positional source read offset",
            })?;
        let bytes_read = source_file.read_at(
            &mut destination_bytes[consumed_byte_count..],
            read_offset_bytes,
        )?;
        if bytes_read == 0 {
            return Err(AlignedExpertPackError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "aligned expert source ended before its declared tensor payload",
            )));
        }
        consumed_byte_count = consumed_byte_count.checked_add(bytes_read).ok_or(
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "advance a positional source read cursor",
            },
        )?;
    }
    Ok(())
}

fn write_all_at(
    destination_file: &File,
    source_bytes: &[u8],
    destination_offset_bytes: u64,
) -> Result<(), AlignedExpertPackError> {
    let mut written_byte_count = 0_usize;
    while written_byte_count < source_bytes.len() {
        let write_offset_bytes = destination_offset_bytes
            .checked_add(u64::try_from(written_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert a positional pack write offset",
                }
            })?)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate a positional pack write offset",
            })?;
        let bytes_written =
            destination_file.write_at(&source_bytes[written_byte_count..], write_offset_bytes)?;
        if bytes_written == 0 {
            return Err(AlignedExpertPackError::Io(std::io::Error::new(
                std::io::ErrorKind::WriteZero,
                "aligned expert pack destination accepted zero positional bytes",
            )));
        }
        written_byte_count = written_byte_count.checked_add(bytes_written).ok_or(
            AlignedExpertPackError::ArithmeticOverflow {
                operation: "advance a positional pack write cursor",
            },
        )?;
    }
    Ok(())
}
