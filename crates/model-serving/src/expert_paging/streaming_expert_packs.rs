//! Per-expert `.apack` streaming sources for the expert pager (on-disk format
//! version 3, one file per `(layer, expert)`).
//!
//! # Why this exists
//!
//! A converted streaming-model revision stores every expert's slice in its own
//! file beside a `manifest.json` describing the whole inventory. The standard
//! pager builds page manifests from SafeTensors shard ranges, so converted
//! revisions would stream through the shards and leave the `.apack` files as
//! unused cargo. This module family supplies the missing wire: when the model
//! directory carries a valid streaming manifest, routed expert pages load
//! through the per-expert files instead.
//!
//! The family splits by responsibility:
//! - this module: manifest detection, inventory validation, and the shared
//!   on-disk layout contract;
//! - [`crate::expert_paging::streaming_expert_pack_plans`]: layer-plan
//!   construction from pack headers for shard-less revisions;
//! - [`crate::expert_paging::streaming_expert_pack_pages`]: routed page
//!   manifests over pack files plus GPU-side assembly.
//!
//! # Why segment offsets are computed, not parsed
//!
//! The per-expert pack layout is fully deterministic: a 64 KiB header region
//! followed by one 64 KiB-aligned segment per ordered tensor. The byte
//! geometry is therefore derivable from the startup-validated layer plan
//! alone. Detection validates the manifest inventory and every file size
//! against that computation plus one spot-checked header, and page builds
//! then trust the computed offsets.
//!
//! # Failure mode
//!
//! Active only when `manifest.json` (format version 3) exists in the model
//! directory. A missing manifest falls back to the shard path unchanged; a
//! present but inconsistent manifest fails the model load loudly rather than
//! silently streaming from a half-converted directory.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::quantized_expert_validation::validate_expert_ids;
use crate::expert_paging::QuantizedExpertLayerPlan;

/// Segment alignment and header size of the per-expert pack layout. These
/// constants mirror the preparer's on-disk contract; they must not diverge.
pub(crate) const PACK_SEGMENT_ALIGNMENT_BYTES: u64 = 64 * 1024;
pub(crate) const PACK_HEADER_BYTES: u64 = PACK_SEGMENT_ALIGNMENT_BYTES;
pub(crate) const PACK_MAGIC: [u8; 8] = *b"ASTEPR01";
pub(crate) const STREAMING_MANIFEST_FORMAT_VERSION: u32 = 3;

/// Minimal manifest view used for detection. Unknown fields are ignored so
/// this reader stays decoupled from the experimental crate's full schema.
#[derive(Deserialize)]
pub(crate) struct StreamingManifestProbe {
    pub(crate) format_version: u32,
    pub(crate) expert_capacity: usize,
    pub(crate) expert_files: Vec<StreamingManifestExpertFile>,
}

#[derive(Deserialize)]
pub(crate) struct StreamingManifestExpertFile {
    pub(crate) layer_index: usize,
    pub(crate) expert_id: usize,
    pub(crate) file_name: String,
    pub(crate) expected_file_byte_count: u64,
}

/// Minimal per-expert pack header view used to build layer plans without
/// SafeTensors shards. Field names mirror the preparer's on-disk contract.
#[derive(Deserialize)]
pub(crate) struct PackHeaderProbe {
    pub(crate) format_version: u32,
    pub(crate) layer_index: usize,
    pub(crate) expert_id: usize,
    pub(crate) expert_capacity: usize,
    // Defaults keep ungated identity-only header reads parsing; the direct-MLX
    // plan builder independently fails closed when geometry is missing.
    #[cfg(feature = "direct-mlx")]
    #[serde(default)]
    pub(crate) quantization_bits: i32,
    #[cfg(feature = "direct-mlx")]
    #[serde(default)]
    pub(crate) quantization_group_size: i32,
    #[cfg(feature = "direct-mlx")]
    #[serde(default)]
    pub(crate) tensor_descriptors: Vec<PackTensorDescriptorProbe>,
}

#[derive(Deserialize)]
#[cfg(feature = "direct-mlx")]
pub(crate) struct PackTensorDescriptorProbe {
    pub(crate) tensor_name: String,
    pub(crate) projection_name: String,
    pub(crate) parameter_name: String,
    pub(crate) dtype_name: String,
    pub(crate) expert_local_shape: Vec<usize>,
    pub(crate) bytes_per_expert: usize,
    pub(crate) pack_segment_offset_bytes: u64,
}

/// Per-expert pack sources for one converted model directory. Indexed exactly
/// like the pager's layer plans: decoder layers first; an appended MTP layer
/// (when present) has no streaming sources and keeps the shard path.
#[derive(Clone, Debug)]
pub struct StreamingExpertPackSources {
    /// One entry per decoder layer, index == decoder layer index. Each entry
    /// holds one pack file path per expert ID.
    expert_file_paths_by_layer: Vec<Vec<PathBuf>>,
}

impl StreamingExpertPackSources {
    /// Number of decoder layers covered by streaming sources.
    #[must_use]
    pub fn decoder_layer_count(&self) -> usize {
        self.expert_file_paths_by_layer.len()
    }

    /// Pack file paths for one decoder layer, indexed by expert ID.
    #[must_use]
    pub fn expert_file_paths(&self, layer_index: usize) -> &[PathBuf] {
        &self.expert_file_paths_by_layer[layer_index]
    }
}

/// Detects a valid streaming revision in `model_dir`. Returns `Ok(None)` when
/// no manifest exists, and a typed error when a manifest exists but does not
/// match the validated layer-plan geometry.
pub fn detect_streaming_expert_pack_sources(
    model_dir: &Path,
    layer_plans: &[QuantizedExpertLayerPlan],
) -> Result<Option<StreamingExpertPackSources>, StreamingExpertPackError> {
    let manifest_path = model_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest = read_streaming_manifest_probe(&manifest_path)?;
    if manifest.format_version != STREAMING_MANIFEST_FORMAT_VERSION {
        return Err(StreamingExpertPackError::UnsupportedFormatVersion {
            manifest_path,
            actual_format_version: manifest.format_version,
        });
    }

    let mut manifest_layer_count = 0_usize;
    let mut declared_files_by_layer = Vec::<Vec<(usize, PathBuf, u64)>>::new();
    for expert_file in &manifest.expert_files {
        while declared_files_by_layer.len() <= expert_file.layer_index {
            declared_files_by_layer.push(Vec::new());
        }
        declared_files_by_layer[expert_file.layer_index].push((
            expert_file.expert_id,
            model_dir.join(&expert_file.file_name),
            expert_file.expected_file_byte_count,
        ));
        manifest_layer_count = manifest_layer_count.max(expert_file.layer_index + 1);
    }
    if manifest_layer_count > layer_plans.len() {
        return Err(StreamingExpertPackError::ManifestLayerCountDisagreement {
            manifest_layer_count,
            decoder_layer_count: layer_plans.len(),
        });
    }

    let mut expert_file_paths_by_layer = Vec::with_capacity(manifest_layer_count);
    for layer_index in 0..manifest_layer_count {
        let layer_plan = &layer_plans[layer_index];
        if manifest.expert_capacity != layer_plan.expert_capacity {
            return Err(StreamingExpertPackError::PackHeaderGeometry {
                description: format!(
                    "the manifest declares capacity {} but layer {layer_index} plans capacity {}",
                    manifest.expert_capacity, layer_plan.expert_capacity
                ),
            });
        }
        let computed_file_byte_count = compute_expert_file_byte_count(layer_plan)?;
        expert_file_paths_by_layer.push(validated_layer_expert_paths(
            layer_index,
            &declared_files_by_layer[layer_index],
            layer_plan.expert_capacity,
            computed_file_byte_count,
        )?);
    }

    // One cheap spot-check binds the computed geometry to real on-disk bytes:
    // the first expert file of layer 0 must carry the pack magic and a header
    // whose declared identity matches the validated plan.
    verify_pack_header(
        &expert_file_paths_by_layer[0][0],
        0,
        0,
        layer_plans[0].expert_capacity,
    )?;

    Ok(Some(StreamingExpertPackSources {
        expert_file_paths_by_layer,
    }))
}

/// Reads and parses one streaming manifest probe.
pub(crate) fn read_streaming_manifest_probe(
    manifest_path: &Path,
) -> Result<StreamingManifestProbe, StreamingExpertPackError> {
    let manifest_text = fs::read_to_string(manifest_path).map_err(|source| {
        StreamingExpertPackError::ManifestUnreadable {
            manifest_path: manifest_path.to_path_buf(),
            source,
        }
    })?;
    serde_json::from_str(&manifest_text).map_err(|source| {
        StreamingExpertPackError::ManifestUnparseable {
            manifest_path: manifest_path.to_path_buf(),
            source,
        }
    })
}

/// Completes one layer's expert inventory: every expert ID must be declared
/// exactly once, its declared byte count must equal the plan-derived file
/// geometry, and the file on disk must match the declaration. A declared size
/// alone proves nothing about a truncated or missing file.
pub(crate) fn validated_layer_expert_paths(
    layer_index: usize,
    declared_files: &[(usize, PathBuf, u64)],
    expert_capacity: usize,
    computed_file_byte_count: u64,
) -> Result<Vec<PathBuf>, StreamingExpertPackError> {
    let mut declared_paths = vec![None; expert_capacity];
    for (expert_id, expert_file_path, expected_file_byte_count) in declared_files {
        if *expert_id >= expert_capacity {
            return Err(StreamingExpertPackError::ExpertIdBeyondCapacity {
                layer_index,
                expert_id: *expert_id,
                expert_capacity,
            });
        }
        declared_paths[*expert_id] = Some((expert_file_path.clone(), *expected_file_byte_count));
    }
    let mut layer_paths = Vec::with_capacity(expert_capacity);
    for (expert_id, declared_file) in declared_paths.into_iter().enumerate() {
        let (expert_file_path, expected_file_byte_count) =
            declared_file.ok_or(StreamingExpertPackError::MissingExpertFile {
                layer_index,
                expert_id,
            })?;
        if expected_file_byte_count != computed_file_byte_count {
            return Err(StreamingExpertPackError::ExpertFileSizeDisagreement {
                expert_file_path,
                expected_file_byte_count,
                computed_file_byte_count,
            });
        }
        let actual_file_byte_count = fs::metadata(&expert_file_path)
            .map_err(|source| StreamingExpertPackError::ExpertFileOpen {
                expert_file_path: expert_file_path.clone(),
                source,
            })?
            .len();
        if actual_file_byte_count != expected_file_byte_count {
            return Err(StreamingExpertPackError::ExpertFileSizeDisagreement {
                expert_file_path,
                expected_file_byte_count: actual_file_byte_count,
                computed_file_byte_count,
            });
        }
        layer_paths.push(expert_file_path);
    }
    Ok(layer_paths)
}

/// Full byte count of one per-expert pack file: 64 KiB header plus one
/// 64 KiB-aligned segment per ordered tensor.
pub(crate) fn compute_expert_file_byte_count(
    layer_plan: &QuantizedExpertLayerPlan,
) -> Result<u64, StreamingExpertPackError> {
    let mut file_byte_count = PACK_HEADER_BYTES;
    for tensor_source in &layer_plan.tensor_sources {
        file_byte_count = file_byte_count
            .checked_add(align_up(
                tensor_source.bytes_per_expert as u64,
                PACK_SEGMENT_ALIGNMENT_BYTES,
            ))
            .ok_or(StreamingExpertPackError::ArithmeticOverflow)?;
    }
    Ok(file_byte_count)
}

pub(crate) fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

/// Reads the 64 KiB header of one pack and validates the magic plus declared
/// identity fields. This binds the file bytes to the format even though read
/// offsets are computed.
pub(crate) fn verify_pack_header(
    expert_file_path: &Path,
    layer_index: usize,
    expert_id: usize,
    expert_capacity: usize,
) -> Result<(), StreamingExpertPackError> {
    let probe = read_pack_header_probe(expert_file_path)?;
    if probe.format_version != STREAMING_MANIFEST_FORMAT_VERSION
        || probe.layer_index != layer_index
        || probe.expert_id != expert_id
        || probe.expert_capacity != expert_capacity
    {
        return Err(StreamingExpertPackError::PackHeaderIdentity {
            expert_file_path: expert_file_path.to_path_buf(),
        });
    }
    Ok(())
}

/// Reads and parses one pack header with the full probe schema.
pub(crate) fn read_pack_header_probe(
    expert_file_path: &Path,
) -> Result<PackHeaderProbe, StreamingExpertPackError> {
    let header_bytes = read_exact_prefix(expert_file_path, PACK_HEADER_BYTES as usize)?;
    if header_bytes[..8] != PACK_MAGIC {
        return Err(StreamingExpertPackError::PackHeaderMagic {
            expert_file_path: expert_file_path.to_path_buf(),
        });
    }
    let header_payload_byte_count = usize::try_from(u64::from_le_bytes(
        header_bytes[8..16].try_into().expect("fixed-width slice"),
    ))
    .map_err(|_| StreamingExpertPackError::PackHeaderPayload {
        expert_file_path: expert_file_path.to_path_buf(),
    })?;
    if header_payload_byte_count == 0 || header_payload_byte_count > PACK_HEADER_BYTES as usize {
        return Err(StreamingExpertPackError::PackHeaderPayload {
            expert_file_path: expert_file_path.to_path_buf(),
        });
    }
    serde_json::from_slice(&header_bytes[16..16 + header_payload_byte_count]).map_err(|source| {
        StreamingExpertPackError::PackHeaderUnparseable {
            expert_file_path: expert_file_path.to_path_buf(),
            source,
        }
    })
}

pub(crate) fn read_exact_prefix(
    expert_file_path: &Path,
    prefix_byte_count: usize,
) -> Result<Vec<u8>, StreamingExpertPackError> {
    let mut buffer = vec![0_u8; prefix_byte_count];
    let mut source_file = fs::File::open(expert_file_path).map_err(|source| {
        StreamingExpertPackError::ExpertFileOpen {
            expert_file_path: expert_file_path.to_path_buf(),
            source,
        }
    })?;
    source_file.read_exact(&mut buffer).map_err(|source| {
        StreamingExpertPackError::ExpertFileRead {
            expert_file_path: expert_file_path.to_path_buf(),
            source,
        }
    })?;
    Ok(buffer)
}

pub(crate) fn validate_routed_expert_ids(
    expert_ids: &[usize],
    expert_capacity: usize,
) -> Result<Vec<usize>, StreamingExpertPackError> {
    validate_expert_ids(expert_ids, expert_capacity).map_err(|error| {
        StreamingExpertPackError::ExpertIdValidation {
            description: error.to_string(),
        }
    })
}

/// Typed failures for streaming expert pack detection and page assembly.
#[derive(Debug, thiserror::Error)]
pub enum StreamingExpertPackError {
    #[error("streaming manifest {manifest_path:?} could not be read: {source}")]
    ManifestUnreadable {
        manifest_path: PathBuf,
        source: std::io::Error,
    },
    #[error("streaming manifest {manifest_path:?} is not valid JSON: {source}")]
    ManifestUnparseable {
        manifest_path: PathBuf,
        source: serde_json::Error,
    },
    #[error(
        "streaming manifest {manifest_path:?} declares unsupported format version {actual_format_version} (expected {STREAMING_MANIFEST_FORMAT_VERSION})"
    )]
    UnsupportedFormatVersion {
        manifest_path: PathBuf,
        actual_format_version: u32,
    },
    #[error(
        "streaming manifest covers {manifest_layer_count} layers but the model declares {decoder_layer_count} decoder layers"
    )]
    ManifestLayerCountDisagreement {
        manifest_layer_count: usize,
        decoder_layer_count: usize,
    },
    #[error("streaming pack header geometry disagrees with the converted plan: {description}")]
    PackHeaderGeometry { description: String },
    #[error(
        "streaming manifest declares layer {layer_index} expert {expert_id} beyond capacity {expert_capacity}"
    )]
    ExpertIdBeyondCapacity {
        layer_index: usize,
        expert_id: usize,
        expert_capacity: usize,
    },
    #[error("streaming manifest is missing the file for layer {layer_index} expert {expert_id}")]
    MissingExpertFile {
        layer_index: usize,
        expert_id: usize,
    },
    #[error(
        "expert pack {expert_file_path:?} byte count {expected_file_byte_count} disagrees with plan geometry {computed_file_byte_count}"
    )]
    ExpertFileSizeDisagreement {
        expert_file_path: PathBuf,
        expected_file_byte_count: u64,
        computed_file_byte_count: u64,
    },
    #[error("expert pack {expert_file_path:?} does not start with the per-expert pack magic")]
    PackHeaderMagic { expert_file_path: PathBuf },
    #[error("expert pack {expert_file_path:?} header payload length is outside the header region")]
    PackHeaderPayload { expert_file_path: PathBuf },
    #[error("expert pack {expert_file_path:?} header payload is not valid JSON: {source}")]
    PackHeaderUnparseable {
        expert_file_path: PathBuf,
        source: serde_json::Error,
    },
    #[error("expert pack {expert_file_path:?} header identity disagrees with the validated plan")]
    PackHeaderIdentity { expert_file_path: PathBuf },
    #[error("expert pack {expert_file_path:?} could not be opened: {source}")]
    ExpertFileOpen {
        expert_file_path: PathBuf,
        source: std::io::Error,
    },
    #[error("expert pack {expert_file_path:?} could not be read: {source}")]
    ExpertFileRead {
        expert_file_path: PathBuf,
        source: std::io::Error,
    },
    #[error("streamed page geometry overflowed")]
    ArithmeticOverflow,
    #[error("streamed expert page rejected routed expert IDs: {description}")]
    ExpertIdValidation { description: String },
    #[error("streamed page is missing {canonical_tensor_name:?} slot {page_slot}")]
    MissingAssembledTensor {
        canonical_tensor_name: String,
        page_slot: usize,
    },
    #[error("concatenating streamed page tensor {canonical_tensor_name:?} failed: {description}")]
    ArrayConcatenation {
        canonical_tensor_name: String,
        description: String,
    },
}
