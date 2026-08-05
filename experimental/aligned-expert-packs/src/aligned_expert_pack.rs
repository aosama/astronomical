use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use astronomical_model_serving::{QuantizationMode, QuantizedExpertLayerPlan};

use crate::{
    aligned_expert_pack_layout::{
        descriptor_from_source, ordered_tensor_sources, validate_segment_extent,
        validate_source_file_range,
    },
    aligned_expert_pack_positional_io::{
        compare_source_tensor_to_pack, copy_source_tensor_to_pack, write_header_region,
    },
};

pub const ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES: u64 = 64 * 1024;
pub(super) const ALIGNED_EXPERT_PACK_HEADER_BYTES: u64 =
    ALIGNED_EXPERT_PACK_SEGMENT_ALIGNMENT_BYTES;
pub(super) const ALIGNED_EXPERT_PACK_MAGIC: [u8; 8] = *b"ASTEPR01";
pub(super) const ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES: usize = 16;

/// Input identity for a deterministic one-layer experimental expert pack.
pub struct AlignedExpertPackBuildRequest<'layer_plan> {
    pub model_id: &'layer_plan str,
    pub model_revision: &'layer_plan str,
    pub layer_index: usize,
    pub layer_plan: &'layer_plan QuantizedExpertLayerPlan,
}

/// Parsed self-describing metadata for an aligned expert pack.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlignedExpertPackHeader {
    pub format_version: u32,
    pub header_payload_byte_count: u64,
    pub model_id: String,
    pub model_revision: String,
    pub layer_index: usize,
    pub layer_prefix: String,
    pub expert_capacity: usize,
    pub quantization_mode: String,
    pub quantization_bits: i32,
    pub quantization_group_size: i32,
    pub tensor_descriptors: Vec<AlignedExpertPackTensorDescriptor>,
    pub expected_pack_byte_count: u64,
}

/// One complete tensor-major segment inside an aligned expert pack.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AlignedExpertPackTensorDescriptor {
    pub tensor_name: String,
    pub projection_name: String,
    pub parameter_name: String,
    pub dtype_name: String,
    pub full_shape: Vec<usize>,
    pub source_file_name: String,
    pub source_file_size_bytes: u64,
    pub source_file_modified_unix_nanoseconds: u64,
    pub source_payload_offset_bytes: u64,
    pub bytes_per_expert: usize,
    pub pack_segment_offset_bytes: u64,
    pub logical_byte_count: usize,
    pub padded_segment_byte_count: u64,
}

/// Bounded pack construction and validation failures.
#[derive(Debug, Error)]
pub enum AlignedExpertPackError {
    #[error("aligned expert pack has an invalid magic prefix")]
    InvalidMagic,
    #[error("aligned expert pack format version {actual_format_version} is unsupported")]
    UnsupportedFormatVersion { actual_format_version: u32 },
    #[error("aligned expert pack header payload has invalid JSON: {0}")]
    InvalidHeaderJson(#[from] serde_json::Error),
    #[error("aligned expert pack header payload is too large: {header_payload_byte_count} bytes")]
    HeaderPayloadTooLarge { header_payload_byte_count: u64 },
    #[error(
        "aligned expert pack header payload length differs: declared {declared_header_payload_byte_count}, actual {actual_header_payload_byte_count}"
    )]
    HeaderPayloadLengthMismatch {
        declared_header_payload_byte_count: u64,
        actual_header_payload_byte_count: u64,
    },
    #[error("aligned expert pack is for model {actual_model_id:?}, not {expected_model_id:?}")]
    ForeignModelId {
        expected_model_id: String,
        actual_model_id: String,
    },
    #[error(
        "aligned expert pack is for model revision {actual_model_revision:?}, not {expected_model_revision:?}"
    )]
    ForeignModelRevision {
        expected_model_revision: String,
        actual_model_revision: String,
    },
    #[error(
        "aligned expert pack layer differs: expected {expected_layer_index} {expected_layer_prefix:?}, got {actual_layer_index} {actual_layer_prefix:?}"
    )]
    ForeignLayer {
        expected_layer_index: usize,
        expected_layer_prefix: String,
        actual_layer_index: usize,
        actual_layer_prefix: String,
    },
    #[error("aligned expert pack quantization contract differs from the validated layer plan")]
    ForeignQuantizationContract,
    #[error(
        "aligned expert pack descriptor count differs: expected {expected_descriptor_count}, got {actual_descriptor_count}"
    )]
    DescriptorCountMismatch {
        expected_descriptor_count: usize,
        actual_descriptor_count: usize,
    },
    #[error("aligned expert pack descriptor {tensor_name:?} differs from the validated layer plan")]
    ForeignTensorDescriptor { tensor_name: String },
    #[error("aligned expert pack descriptor {tensor_name:?} is not 64 KB aligned")]
    UnalignedSegmentOffset { tensor_name: String },
    #[error("aligned expert pack descriptor {tensor_name:?} has invalid padded extent")]
    InvalidPaddedSegmentExtent { tensor_name: String },
    #[error("aligned expert pack descriptor {tensor_name:?} overlaps the previous segment")]
    OverlappingSegment { tensor_name: String },
    #[error("aligned expert pack arithmetic overflowed while {operation}")]
    ArithmeticOverflow { operation: &'static str },
    #[error("aligned expert pack source tensor {tensor_name:?} exceeds its source file")]
    SourceRangeExceedsFile { tensor_name: String },
    #[error(
        "aligned expert pack source file modification identity changed for tensor {tensor_name:?}"
    )]
    SourceFileModificationIdentityMismatch { tensor_name: String },
    #[error("aligned expert pack source tensor {tensor_name:?} has an invalid shape byte count")]
    InvalidTensorShapeByteCount { tensor_name: String },
    #[error("aligned expert pack output already exists at {pack_output_path:?}")]
    OutputAlreadyExists { pack_output_path: PathBuf },
    #[error(
        "aligned expert pack file length differs: expected {expected_pack_byte_count}, got {actual_pack_byte_count}"
    )]
    PackLengthMismatch {
        expected_pack_byte_count: u64,
        actual_pack_byte_count: u64,
    },
    #[error(
        "aligned expert pack payload differs from source tensor {tensor_name:?} at tensor byte offset {tensor_byte_offset}"
    )]
    PayloadByteMismatch {
        tensor_name: String,
        tensor_byte_offset: u64,
    },
    #[error("aligned expert pack I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Creates and verifies one aligned expert pack without materializing a layer-sized byte vector.
pub fn build_aligned_expert_pack(
    pack_output_path: &Path,
    build_request: &AlignedExpertPackBuildRequest<'_>,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    if pack_output_path.exists() {
        return Err(AlignedExpertPackError::OutputAlreadyExists {
            pack_output_path: pack_output_path.to_path_buf(),
        });
    }
    let parent_directory = pack_output_path.parent().ok_or_else(|| {
        AlignedExpertPackError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "aligned expert pack output must have a parent directory",
        ))
    })?;
    let in_progress_pack_path = parent_directory.join(format!(
        ".{}.building-{}",
        pack_output_path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("aligned-expert-pack"),
        std::process::id()
    ));
    let build_outcome = build_unpublished_aligned_expert_pack(
        &in_progress_pack_path,
        pack_output_path,
        build_request,
    );
    match build_outcome {
        Ok(aligned_expert_pack_header) => {
            fs::rename(&in_progress_pack_path, pack_output_path)?;
            Ok(aligned_expert_pack_header)
        }
        Err(build_error) => {
            let _ = fs::remove_file(&in_progress_pack_path);
            Err(build_error)
        }
    }
}

pub(super) fn plan_aligned_expert_pack_header(
    build_request: &AlignedExpertPackBuildRequest<'_>,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    planned_aligned_expert_pack_header(build_request)
}

/// Reads only the fixed pack header region and performs no tensor payload access.
pub fn read_aligned_expert_pack_header(
    aligned_expert_pack_path: &Path,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    let mut aligned_expert_pack_file = File::open(aligned_expert_pack_path)?;
    let mut header_prefix_bytes = [0_u8; ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES];
    aligned_expert_pack_file.read_exact(&mut header_prefix_bytes)?;
    if header_prefix_bytes[..ALIGNED_EXPERT_PACK_MAGIC.len()] != ALIGNED_EXPERT_PACK_MAGIC {
        return Err(AlignedExpertPackError::InvalidMagic);
    }
    let header_payload_byte_count = u64::from_le_bytes(
        header_prefix_bytes[ALIGNED_EXPERT_PACK_MAGIC.len()..]
            .try_into()
            .map_err(|_| {
                AlignedExpertPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "aligned expert pack header length prefix is malformed",
                ))
            })?,
    );
    let maximum_header_payload_bytes = ALIGNED_EXPERT_PACK_HEADER_BYTES
        .checked_sub(ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES as u64)
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate the maximum aligned expert pack header payload",
        })?;
    if header_payload_byte_count > maximum_header_payload_bytes {
        return Err(AlignedExpertPackError::HeaderPayloadTooLarge {
            header_payload_byte_count,
        });
    }
    let mut header_payload_bytes =
        vec![
            0_u8;
            usize::try_from(header_payload_byte_count).map_err(|_| {
                AlignedExpertPackError::ArithmeticOverflow {
                    operation: "convert the aligned expert pack header payload length",
                }
            })?
        ];
    aligned_expert_pack_file.read_exact(&mut header_payload_bytes)?;
    let aligned_expert_pack_header: AlignedExpertPackHeader =
        serde_json::from_slice(&header_payload_bytes)?;
    if aligned_expert_pack_header.format_version != 2 {
        return Err(AlignedExpertPackError::UnsupportedFormatVersion {
            actual_format_version: aligned_expert_pack_header.format_version,
        });
    }
    if aligned_expert_pack_header.header_payload_byte_count != header_payload_byte_count {
        return Err(AlignedExpertPackError::HeaderPayloadLengthMismatch {
            declared_header_payload_byte_count: aligned_expert_pack_header
                .header_payload_byte_count,
            actual_header_payload_byte_count: header_payload_byte_count,
        });
    }
    Ok(aligned_expert_pack_header)
}

/// Validates a parsed pack against the current startup-validated layer plan.
pub fn validate_aligned_expert_pack_header(
    aligned_expert_pack_path: &Path,
    aligned_expert_pack_header: &AlignedExpertPackHeader,
    expected_layer_plan: &QuantizedExpertLayerPlan,
    expected_model_id: &str,
    expected_model_revision: &str,
    expected_layer_index: usize,
) -> Result<(), AlignedExpertPackError> {
    if aligned_expert_pack_header.model_id != expected_model_id {
        return Err(AlignedExpertPackError::ForeignModelId {
            expected_model_id: expected_model_id.to_owned(),
            actual_model_id: aligned_expert_pack_header.model_id.clone(),
        });
    }
    if aligned_expert_pack_header.model_revision != expected_model_revision {
        return Err(AlignedExpertPackError::ForeignModelRevision {
            expected_model_revision: expected_model_revision.to_owned(),
            actual_model_revision: aligned_expert_pack_header.model_revision.clone(),
        });
    }
    if aligned_expert_pack_header.layer_index != expected_layer_index
        || aligned_expert_pack_header.layer_prefix != expected_layer_plan.layer_prefix
    {
        return Err(AlignedExpertPackError::ForeignLayer {
            expected_layer_index,
            expected_layer_prefix: expected_layer_plan.layer_prefix.clone(),
            actual_layer_index: aligned_expert_pack_header.layer_index,
            actual_layer_prefix: aligned_expert_pack_header.layer_prefix.clone(),
        });
    }
    if aligned_expert_pack_header.expert_capacity != expected_layer_plan.expert_capacity
        || aligned_expert_pack_header.quantization_mode
            != quantization_mode_name(expected_layer_plan)
        || aligned_expert_pack_header.quantization_bits != expected_layer_plan.quantization_bits
        || aligned_expert_pack_header.quantization_group_size
            != expected_layer_plan.quantization_group_size
    {
        return Err(AlignedExpertPackError::ForeignQuantizationContract);
    }
    let expected_tensor_sources = ordered_tensor_sources(expected_layer_plan)?;
    if aligned_expert_pack_header.tensor_descriptors.len() != expected_tensor_sources.len() {
        return Err(AlignedExpertPackError::DescriptorCountMismatch {
            expected_descriptor_count: expected_tensor_sources.len(),
            actual_descriptor_count: aligned_expert_pack_header.tensor_descriptors.len(),
        });
    }
    let mut expected_segment_offset_bytes = ALIGNED_EXPERT_PACK_HEADER_BYTES;
    for (tensor_descriptor, tensor_source) in aligned_expert_pack_header
        .tensor_descriptors
        .iter()
        .zip(expected_tensor_sources)
    {
        let expected_tensor_descriptor = descriptor_from_source(
            tensor_source,
            expected_segment_offset_bytes,
            expected_layer_plan.expert_capacity,
        )?;
        if tensor_descriptor.source_file_modified_unix_nanoseconds
            != expected_tensor_descriptor.source_file_modified_unix_nanoseconds
        {
            return Err(
                AlignedExpertPackError::SourceFileModificationIdentityMismatch {
                    tensor_name: tensor_descriptor.tensor_name.clone(),
                },
            );
        }
        if tensor_descriptor != &expected_tensor_descriptor {
            return Err(AlignedExpertPackError::ForeignTensorDescriptor {
                tensor_name: tensor_descriptor.tensor_name.clone(),
            });
        }
        validate_segment_extent(tensor_descriptor, expected_segment_offset_bytes)?;
        expected_segment_offset_bytes = tensor_descriptor
            .pack_segment_offset_bytes
            .checked_add(tensor_descriptor.padded_segment_byte_count)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "advance an aligned expert pack segment offset",
            })?;
    }
    if expected_segment_offset_bytes != aligned_expert_pack_header.expected_pack_byte_count {
        return Err(AlignedExpertPackError::PackLengthMismatch {
            expected_pack_byte_count: expected_segment_offset_bytes,
            actual_pack_byte_count: aligned_expert_pack_header.expected_pack_byte_count,
        });
    }
    let actual_pack_byte_count = fs::metadata(aligned_expert_pack_path)?.len();
    if actual_pack_byte_count != aligned_expert_pack_header.expected_pack_byte_count {
        return Err(AlignedExpertPackError::PackLengthMismatch {
            expected_pack_byte_count: aligned_expert_pack_header.expected_pack_byte_count,
            actual_pack_byte_count,
        });
    }
    Ok(())
}

/// Compares every logical pack byte with its validated source tensor.
pub fn validate_aligned_expert_pack_payload(
    aligned_expert_pack_path: &Path,
    aligned_expert_pack_header: &AlignedExpertPackHeader,
    expected_layer_plan: &QuantizedExpertLayerPlan,
) -> Result<(), AlignedExpertPackError> {
    let aligned_expert_pack_file = File::open(aligned_expert_pack_path)?;
    let expected_tensor_sources = ordered_tensor_sources(expected_layer_plan)?;
    for (tensor_source, tensor_descriptor) in expected_tensor_sources
        .iter()
        .zip(&aligned_expert_pack_header.tensor_descriptors)
    {
        compare_source_tensor_to_pack(tensor_source, tensor_descriptor, &aligned_expert_pack_file)?;
    }
    Ok(())
}

fn build_unpublished_aligned_expert_pack(
    in_progress_pack_path: &Path,
    requested_pack_output_path: &Path,
    build_request: &AlignedExpertPackBuildRequest<'_>,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    let ordered_tensor_sources = ordered_tensor_sources(build_request.layer_plan)?;
    let aligned_expert_pack_header = planned_aligned_expert_pack_header(build_request)?;
    let serialized_header_payload = serde_json::to_vec(&aligned_expert_pack_header)?;
    let in_progress_pack_file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(in_progress_pack_path)?;
    in_progress_pack_file.set_len(aligned_expert_pack_header.expected_pack_byte_count)?;
    write_header_region(&in_progress_pack_file, &serialized_header_payload)?;
    for (tensor_source, tensor_descriptor) in ordered_tensor_sources
        .iter()
        .zip(&aligned_expert_pack_header.tensor_descriptors)
    {
        copy_source_tensor_to_pack(tensor_source, tensor_descriptor, &in_progress_pack_file)?;
    }
    in_progress_pack_file.sync_all()?;
    drop(in_progress_pack_file);
    let reopened_pack_header = read_aligned_expert_pack_header(in_progress_pack_path)?;
    validate_aligned_expert_pack_header(
        in_progress_pack_path,
        &reopened_pack_header,
        build_request.layer_plan,
        build_request.model_id,
        build_request.model_revision,
        build_request.layer_index,
    )?;
    validate_aligned_expert_pack_payload(
        in_progress_pack_path,
        &reopened_pack_header,
        build_request.layer_plan,
    )?;
    if requested_pack_output_path.exists() {
        return Err(AlignedExpertPackError::OutputAlreadyExists {
            pack_output_path: requested_pack_output_path.to_path_buf(),
        });
    }
    Ok(reopened_pack_header)
}

fn planned_aligned_expert_pack_header(
    build_request: &AlignedExpertPackBuildRequest<'_>,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    let ordered_tensor_sources = ordered_tensor_sources(build_request.layer_plan)?;
    let mut tensor_descriptors = Vec::with_capacity(ordered_tensor_sources.len());
    let mut next_segment_offset_bytes = ALIGNED_EXPERT_PACK_HEADER_BYTES;
    for tensor_source in &ordered_tensor_sources {
        let tensor_descriptor = descriptor_from_source(
            tensor_source,
            next_segment_offset_bytes,
            build_request.layer_plan.expert_capacity,
        )?;
        validate_source_file_range(tensor_source, &tensor_descriptor)?;
        next_segment_offset_bytes = tensor_descriptor
            .pack_segment_offset_bytes
            .checked_add(tensor_descriptor.padded_segment_byte_count)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate an aligned expert pack byte length",
            })?;
        tensor_descriptors.push(tensor_descriptor);
    }
    serialize_header_with_stable_length(AlignedExpertPackHeader {
        format_version: 2,
        header_payload_byte_count: 0,
        model_id: build_request.model_id.to_owned(),
        model_revision: build_request.model_revision.to_owned(),
        layer_index: build_request.layer_index,
        layer_prefix: build_request.layer_plan.layer_prefix.clone(),
        expert_capacity: build_request.layer_plan.expert_capacity,
        quantization_mode: quantization_mode_name(build_request.layer_plan).to_owned(),
        quantization_bits: build_request.layer_plan.quantization_bits,
        quantization_group_size: build_request.layer_plan.quantization_group_size,
        tensor_descriptors,
        expected_pack_byte_count: next_segment_offset_bytes,
    })
}

fn serialize_header_with_stable_length(
    mut aligned_expert_pack_header: AlignedExpertPackHeader,
) -> Result<AlignedExpertPackHeader, AlignedExpertPackError> {
    for _header_serialization_attempt in 0..4 {
        let serialized_header_payload = serde_json::to_vec(&aligned_expert_pack_header)?;
        let actual_header_payload_byte_count = u64::try_from(serialized_header_payload.len())
            .map_err(|_| AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert aligned expert pack header payload length",
            })?;
        if actual_header_payload_byte_count == aligned_expert_pack_header.header_payload_byte_count
        {
            let maximum_header_payload_byte_count = ALIGNED_EXPERT_PACK_HEADER_BYTES
                .checked_sub(ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES as u64)
                .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                    operation: "calculate the maximum aligned expert pack header payload",
                })?;
            if actual_header_payload_byte_count > maximum_header_payload_byte_count {
                return Err(AlignedExpertPackError::HeaderPayloadTooLarge {
                    header_payload_byte_count: actual_header_payload_byte_count,
                });
            }
            return Ok(aligned_expert_pack_header);
        }
        aligned_expert_pack_header.header_payload_byte_count = actual_header_payload_byte_count;
    }
    Err(AlignedExpertPackError::HeaderPayloadLengthMismatch {
        declared_header_payload_byte_count: aligned_expert_pack_header.header_payload_byte_count,
        actual_header_payload_byte_count: u64::try_from(
            serde_json::to_vec(&aligned_expert_pack_header)?.len(),
        )
        .unwrap_or(u64::MAX),
    })
}

fn quantization_mode_name(layer_plan: &QuantizedExpertLayerPlan) -> &'static str {
    match layer_plan.quantization_mode {
        QuantizationMode::Affine => "affine",
        QuantizationMode::NativeBfloat16 => "native_bfloat16",
    }
}
