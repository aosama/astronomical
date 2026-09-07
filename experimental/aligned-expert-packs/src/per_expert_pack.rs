//! One-file-per-expert aligned pack (on-disk format version 3).
//!
//! Issue #430 described this generation as "v2" relative to the layer-major
//! research pack. That layer-major pack already occupies format_version 2, so
//! these per-expert files use format_version 3 and cannot be mistaken for it.

use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use astronomical_model_serving::{QuantizationMode, QuantizedExpertLayerPlan};

use crate::aligned_expert_pack::{
    ALIGNED_EXPERT_PACK_HEADER_BYTES, ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES,
    ALIGNED_EXPERT_PACK_MAGIC, AlignedExpertPackError,
};
use crate::aligned_expert_pack_layout::ordered_tensor_sources;
use crate::aligned_expert_pack_positional_io::{
    compare_file_range_to_pack, copy_file_range_to_pack, write_header_region,
};
use crate::per_expert_pack_layout::{
    descriptor_from_source_for_expert, validate_source_file_range_for_expert,
};

/// On-disk format version for one-file-per-expert packs.
pub const PER_EXPERT_PACK_FORMAT_VERSION: u32 = 3;

/// Input identity for one expert file inside one layer.
pub struct PerExpertPackBuildRequest<'layer_plan> {
    pub model_id: &'layer_plan str,
    pub model_revision: &'layer_plan str,
    pub layer_index: usize,
    pub expert_id: usize,
    pub layer_plan: &'layer_plan QuantizedExpertLayerPlan,
}

/// Parsed self-describing metadata for one per-expert pack file.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PerExpertPackHeader {
    pub format_version: u32,
    pub header_payload_byte_count: u64,
    pub model_id: String,
    pub model_revision: String,
    pub layer_index: usize,
    pub layer_prefix: String,
    pub expert_id: usize,
    pub expert_capacity: usize,
    pub quantization_mode: String,
    pub quantization_bits: i32,
    pub quantization_group_size: i32,
    pub tensor_descriptors: Vec<PerExpertPackTensorDescriptor>,
    pub expected_file_byte_count: u64,
}

pub use crate::per_expert_pack_layout::PerExpertPackTensorDescriptor;

/// Creates and verifies one per-expert pack without decoding tensor values.
pub fn build_per_expert_pack(
    pack_output_path: &Path,
    build_request: &PerExpertPackBuildRequest<'_>,
) -> Result<PerExpertPackHeader, AlignedExpertPackError> {
    if pack_output_path.exists() {
        return Err(AlignedExpertPackError::OutputAlreadyExists {
            pack_output_path: pack_output_path.to_path_buf(),
        });
    }
    let parent_directory = pack_output_path.parent().ok_or_else(|| {
        AlignedExpertPackError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "per-expert pack output must have a parent directory",
        ))
    })?;
    let in_progress_pack_path = parent_directory.join(format!(
        ".{}.building-{}",
        pack_output_path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            .unwrap_or("per-expert-pack"),
        std::process::id()
    ));
    let build_outcome =
        build_unpublished_per_expert_pack(&in_progress_pack_path, pack_output_path, build_request);
    match build_outcome {
        Ok(per_expert_pack_header) => {
            fs::rename(&in_progress_pack_path, pack_output_path)?;
            Ok(per_expert_pack_header)
        }
        Err(build_error) => {
            let _ = fs::remove_file(&in_progress_pack_path);
            Err(build_error)
        }
    }
}

/// Reads only the fixed pack header region.
pub fn read_per_expert_pack_header(
    per_expert_pack_path: &Path,
) -> Result<PerExpertPackHeader, AlignedExpertPackError> {
    let mut per_expert_pack_file = File::open(per_expert_pack_path)?;
    let mut header_prefix_bytes = [0_u8; ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES];
    per_expert_pack_file.read_exact(&mut header_prefix_bytes)?;
    if header_prefix_bytes[..ALIGNED_EXPERT_PACK_MAGIC.len()] != ALIGNED_EXPERT_PACK_MAGIC {
        return Err(AlignedExpertPackError::InvalidMagic);
    }
    let header_payload_byte_count = u64::from_le_bytes(
        header_prefix_bytes[ALIGNED_EXPERT_PACK_MAGIC.len()..]
            .try_into()
            .map_err(|_| {
                AlignedExpertPackError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "per-expert pack header length prefix is malformed",
                ))
            })?,
    );
    let maximum_header_payload_bytes = ALIGNED_EXPERT_PACK_HEADER_BYTES
        .checked_sub(ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES as u64)
        .ok_or(AlignedExpertPackError::ArithmeticOverflow {
            operation: "calculate the maximum per-expert pack header payload",
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
                    operation: "convert the per-expert pack header payload length",
                }
            })?
        ];
    per_expert_pack_file.read_exact(&mut header_payload_bytes)?;
    let per_expert_pack_header: PerExpertPackHeader =
        serde_json::from_slice(&header_payload_bytes)?;
    if per_expert_pack_header.format_version != PER_EXPERT_PACK_FORMAT_VERSION {
        return Err(AlignedExpertPackError::UnsupportedFormatVersion {
            actual_format_version: per_expert_pack_header.format_version,
        });
    }
    if per_expert_pack_header.header_payload_byte_count != header_payload_byte_count {
        return Err(AlignedExpertPackError::HeaderPayloadLengthMismatch {
            declared_header_payload_byte_count: per_expert_pack_header.header_payload_byte_count,
            actual_header_payload_byte_count: header_payload_byte_count,
        });
    }
    Ok(per_expert_pack_header)
}

/// Validates a parsed per-expert pack against the layer plan and expert identity.
pub fn validate_per_expert_pack_header(
    per_expert_pack_path: &Path,
    per_expert_pack_header: &PerExpertPackHeader,
    expected_layer_plan: &QuantizedExpertLayerPlan,
    expected_model_id: &str,
    expected_model_revision: &str,
    expected_layer_index: usize,
    expected_expert_id: usize,
) -> Result<(), AlignedExpertPackError> {
    if per_expert_pack_header.model_id != expected_model_id {
        return Err(AlignedExpertPackError::ForeignModelId {
            expected_model_id: expected_model_id.to_owned(),
            actual_model_id: per_expert_pack_header.model_id.clone(),
        });
    }
    if per_expert_pack_header.model_revision != expected_model_revision {
        return Err(AlignedExpertPackError::ForeignModelRevision {
            expected_model_revision: expected_model_revision.to_owned(),
            actual_model_revision: per_expert_pack_header.model_revision.clone(),
        });
    }
    if per_expert_pack_header.layer_index != expected_layer_index
        || per_expert_pack_header.layer_prefix != expected_layer_plan.layer_prefix
    {
        return Err(AlignedExpertPackError::ForeignLayer {
            expected_layer_index,
            expected_layer_prefix: expected_layer_plan.layer_prefix.clone(),
            actual_layer_index: per_expert_pack_header.layer_index,
            actual_layer_prefix: per_expert_pack_header.layer_prefix.clone(),
        });
    }
    if per_expert_pack_header.expert_id != expected_expert_id {
        return Err(AlignedExpertPackError::ForeignExpertId {
            expected_expert_id,
            actual_expert_id: per_expert_pack_header.expert_id,
        });
    }
    if per_expert_pack_header.expert_capacity != expected_layer_plan.expert_capacity
        || per_expert_pack_header.quantization_mode != quantization_mode_name(expected_layer_plan)
        || per_expert_pack_header.quantization_bits != expected_layer_plan.quantization_bits
        || per_expert_pack_header.quantization_group_size
            != expected_layer_plan.quantization_group_size
    {
        return Err(AlignedExpertPackError::ForeignQuantizationContract);
    }
    let expected_header = planned_per_expert_pack_header(
        expected_model_id,
        expected_model_revision,
        expected_layer_index,
        expected_expert_id,
        expected_layer_plan,
    )?;
    if per_expert_pack_header.tensor_descriptors != expected_header.tensor_descriptors {
        let tensor_name = per_expert_pack_header
            .tensor_descriptors
            .first()
            .map(|tensor_descriptor| tensor_descriptor.tensor_name.clone())
            .unwrap_or_default();
        return Err(AlignedExpertPackError::ForeignTensorDescriptor { tensor_name });
    }
    let actual_file_byte_count = fs::metadata(per_expert_pack_path)?.len();
    if actual_file_byte_count != per_expert_pack_header.expected_file_byte_count {
        return Err(AlignedExpertPackError::PackLengthMismatch {
            expected_pack_byte_count: per_expert_pack_header.expected_file_byte_count,
            actual_pack_byte_count: actual_file_byte_count,
        });
    }
    Ok(())
}

/// Compares every packed expert slice with its source tensor slice.
pub fn validate_per_expert_pack_payload(
    per_expert_pack_path: &Path,
    per_expert_pack_header: &PerExpertPackHeader,
    expected_layer_plan: &QuantizedExpertLayerPlan,
) -> Result<(), AlignedExpertPackError> {
    let per_expert_pack_file = File::open(per_expert_pack_path)?;
    let expected_tensor_sources = ordered_tensor_sources(expected_layer_plan)?;
    for (tensor_source, tensor_descriptor) in expected_tensor_sources
        .iter()
        .zip(&per_expert_pack_header.tensor_descriptors)
    {
        compare_file_range_to_pack(
            &tensor_source.source_file,
            tensor_descriptor.source_expert_payload_offset_bytes,
            tensor_descriptor.logical_byte_count,
            &per_expert_pack_file,
            tensor_descriptor.pack_segment_offset_bytes,
            &tensor_descriptor.tensor_name,
        )?;
    }
    Ok(())
}

fn build_unpublished_per_expert_pack(
    in_progress_pack_path: &Path,
    requested_pack_output_path: &Path,
    build_request: &PerExpertPackBuildRequest<'_>,
) -> Result<PerExpertPackHeader, AlignedExpertPackError> {
    let ordered_tensor_sources = ordered_tensor_sources(build_request.layer_plan)?;
    let per_expert_pack_header = planned_per_expert_pack_header(
        build_request.model_id,
        build_request.model_revision,
        build_request.layer_index,
        build_request.expert_id,
        build_request.layer_plan,
    )?;
    let serialized_header_payload = serde_json::to_vec(&per_expert_pack_header)?;
    let in_progress_pack_file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(in_progress_pack_path)?;
    in_progress_pack_file.set_len(per_expert_pack_header.expected_file_byte_count)?;
    write_header_region(&in_progress_pack_file, &serialized_header_payload)?;
    for (tensor_source, tensor_descriptor) in ordered_tensor_sources
        .iter()
        .zip(&per_expert_pack_header.tensor_descriptors)
    {
        copy_file_range_to_pack(
            &tensor_source.source_file,
            tensor_descriptor.source_expert_payload_offset_bytes,
            tensor_descriptor.logical_byte_count,
            &in_progress_pack_file,
            tensor_descriptor.pack_segment_offset_bytes,
        )?;
    }
    in_progress_pack_file.sync_all()?;
    drop(in_progress_pack_file);
    let reopened_pack_header = read_per_expert_pack_header(in_progress_pack_path)?;
    validate_per_expert_pack_header(
        in_progress_pack_path,
        &reopened_pack_header,
        build_request.layer_plan,
        build_request.model_id,
        build_request.model_revision,
        build_request.layer_index,
        build_request.expert_id,
    )?;
    validate_per_expert_pack_payload(
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

fn planned_per_expert_pack_header(
    model_id: &str,
    model_revision: &str,
    layer_index: usize,
    expert_id: usize,
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<PerExpertPackHeader, AlignedExpertPackError> {
    if expert_id >= layer_plan.expert_capacity {
        return Err(AlignedExpertPackError::ForeignExpertId {
            expected_expert_id: expert_id,
            actual_expert_id: layer_plan.expert_capacity,
        });
    }
    let ordered_tensor_sources = ordered_tensor_sources(layer_plan)?;
    let mut tensor_descriptors = Vec::with_capacity(ordered_tensor_sources.len());
    let mut next_segment_offset_bytes = ALIGNED_EXPERT_PACK_HEADER_BYTES;
    for tensor_source in &ordered_tensor_sources {
        let tensor_descriptor = descriptor_from_source_for_expert(
            tensor_source,
            expert_id,
            next_segment_offset_bytes,
            layer_plan.expert_capacity,
        )?;
        validate_source_file_range_for_expert(tensor_source, &tensor_descriptor)?;
        next_segment_offset_bytes = tensor_descriptor
            .pack_segment_offset_bytes
            .checked_add(tensor_descriptor.padded_segment_byte_count)
            .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                operation: "calculate a per-expert pack byte length",
            })?;
        tensor_descriptors.push(tensor_descriptor);
    }
    serialize_header_with_stable_length(PerExpertPackHeader {
        format_version: PER_EXPERT_PACK_FORMAT_VERSION,
        header_payload_byte_count: 0,
        model_id: model_id.to_owned(),
        model_revision: model_revision.to_owned(),
        layer_index,
        layer_prefix: layer_plan.layer_prefix.clone(),
        expert_id,
        expert_capacity: layer_plan.expert_capacity,
        quantization_mode: quantization_mode_name(layer_plan).to_owned(),
        quantization_bits: layer_plan.quantization_bits,
        quantization_group_size: layer_plan.quantization_group_size,
        tensor_descriptors,
        expected_file_byte_count: next_segment_offset_bytes,
    })
}

fn serialize_header_with_stable_length(
    mut per_expert_pack_header: PerExpertPackHeader,
) -> Result<PerExpertPackHeader, AlignedExpertPackError> {
    for _header_serialization_attempt in 0..4 {
        let serialized_header_payload = serde_json::to_vec(&per_expert_pack_header)?;
        let actual_header_payload_byte_count = u64::try_from(serialized_header_payload.len())
            .map_err(|_| AlignedExpertPackError::ArithmeticOverflow {
                operation: "convert per-expert pack header payload length",
            })?;
        if actual_header_payload_byte_count == per_expert_pack_header.header_payload_byte_count {
            let maximum_header_payload_byte_count = ALIGNED_EXPERT_PACK_HEADER_BYTES
                .checked_sub(ALIGNED_EXPERT_PACK_HEADER_PREFIX_BYTES as u64)
                .ok_or(AlignedExpertPackError::ArithmeticOverflow {
                    operation: "calculate the maximum per-expert pack header payload",
                })?;
            if actual_header_payload_byte_count > maximum_header_payload_byte_count {
                return Err(AlignedExpertPackError::HeaderPayloadTooLarge {
                    header_payload_byte_count: actual_header_payload_byte_count,
                });
            }
            return Ok(per_expert_pack_header);
        }
        per_expert_pack_header.header_payload_byte_count = actual_header_payload_byte_count;
    }
    Err(AlignedExpertPackError::HeaderPayloadLengthMismatch {
        declared_header_payload_byte_count: per_expert_pack_header.header_payload_byte_count,
        actual_header_payload_byte_count: u64::try_from(
            serde_json::to_vec(&per_expert_pack_header)?.len(),
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

/// Relative path of one expert file inside a streaming-model revision.
pub fn per_expert_pack_relative_path(layer_index: usize, expert_id: usize) -> PathBuf {
    PathBuf::from("layers")
        .join(layer_index.to_string())
        .join(format!("{expert_id}.apack"))
}
