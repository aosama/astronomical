//! Bounded, MLX-free validator for persisted model-state files.
//!
//! Version 7 derives decoder-state tensor validation from the model-owned,
//! architecture-neutral cache layout. Version 8 invalidates snapshots written
//! before the expert-residency correction. Version 6 used Qwen-specific names.
//! This module reads only the length-prefixed JSON header, validates the declared
//! tensor layout against the expected Qwen3.5-MoE file kind, and returns the
//! metadata the prompt-cache owner needs to decide whether to load the file.
//! It never reads the multi-megabyte tensor payload region.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::decoder_cache::DecoderCachePersistedTensorLayout;
use crate::safetensors::SafetensorsTensorView;

use super::block_format_error::PersistentPromptCacheBlockError;
use super::model_contract::PersistentPromptCacheModelContract;
use super::persistent_safetensors_header::{
    PersistentSafetensorsHeaderError, read_persistent_safetensors_header,
};

/// Current persistent prompt-cache state version. Bump when the on-disk layout
/// or execution math changes in a way that invalidates serialized model state.
///
/// Version 4: full-attention key/value blocks and GatedDeltaNet recurrent
/// snapshots live in separate safetensors files so the fixed recurrent state is
/// not duplicated in every cache block.
/// Version 5: invalidates state produced before stable gated-delta softplus and
/// shape-safe variable-length compiled decay execution.
/// Version 8: invalidates state produced before the complete-layer rollback
/// corrected macOS-pressure expert residency.
/// Version 10: binds model-derived block geometry and the complete storage
/// contract fingerprint while allowing each declared state kind independently.
pub(crate) const PERSISTENT_PROMPT_CACHE_FORMAT_VERSION: &str = "10";

/// Parsed and validated metadata for one Qwen3.5-MoE persistent prompt-cache block on disk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentPromptCacheBlockHeader {
    format_version: String,
    model_id: String,
    model_revision: String,
    storage_contract_fingerprint: String,
    block_token_count: usize,
    tensor_count: usize,
}

impl PersistentPromptCacheBlockHeader {
    /// Reads and validates one full-attention key/value block header without touching the payload.
    pub fn read_kv_block_from_file(
        persistent_prompt_cache_block_file: &File,
        persistent_prompt_cache_block_path: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentPromptCacheBlockError> {
        Self::read_from_file_kind(
            persistent_prompt_cache_block_file,
            persistent_prompt_cache_block_path,
            PersistentPromptCacheFileKind::SequenceStateBlock,
            persistent_prompt_cache_model_contract,
        )
    }

    /// Reads and validates one GatedDeltaNet recurrent snapshot header without touching the payload.
    pub fn read_recurrent_snapshot_from_file(
        persistent_prompt_cache_recurrent_snapshot_file: &File,
        persistent_prompt_cache_recurrent_snapshot_path: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentPromptCacheBlockError> {
        Self::read_from_file_kind(
            persistent_prompt_cache_recurrent_snapshot_file,
            persistent_prompt_cache_recurrent_snapshot_path,
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            persistent_prompt_cache_model_contract,
        )
    }

    fn read_from_file_kind(
        persistent_prompt_cache_block_file: &File,
        persistent_prompt_cache_block_path: &Path,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentPromptCacheBlockError> {
        // Read and validate only bounded header bytes. Startup scanning must not
        // deserialize every tensor payload just to identify usable SSD entries.
        let parsed_header = read_persistent_safetensors_header(
            persistent_prompt_cache_block_file,
            persistent_prompt_cache_block_path,
        )
        .map_err(persistent_safetensors_header_error_to_block_error)?;
        let metadata =
            extract_required_metadata(&parsed_header.metadata, persistent_prompt_cache_block_path)?;
        // Model identity and format validation occur before tensor checks so a
        // directory shared with another model revision remains harmless.
        validate_metadata(
            &metadata,
            persistent_prompt_cache_block_path,
            persistent_prompt_cache_model_contract,
        )?;
        validate_tensor_layout(
            &parsed_header.tensor_views,
            metadata.block_token_count,
            persistent_prompt_cache_file_kind,
            persistent_prompt_cache_block_path,
            persistent_prompt_cache_model_contract,
        )?;
        validate_tensor_offsets(
            &parsed_header.tensor_views,
            parsed_header.data_section_start_bytes,
            parsed_header.file_size_bytes,
            persistent_prompt_cache_block_path,
        )?;
        Ok(Self {
            format_version: metadata.format_version,
            model_id: metadata.model_id,
            model_revision: metadata.model_revision,
            storage_contract_fingerprint: metadata.storage_contract_fingerprint,
            block_token_count: metadata.block_token_count,
            tensor_count: parsed_header.tensor_views.len(),
        })
    }

    /// Returns the on-disk format version string.
    #[must_use]
    pub fn format_version(&self) -> &str {
        &self.format_version
    }

    /// Returns the model id stamped into the block.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the model revision stamped into the block.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// Returns the number of prompt tokens captured by this block.
    #[must_use]
    pub fn block_token_count(&self) -> usize {
        self.block_token_count
    }

    /// Returns the exact model storage-contract fingerprint stamped into the file.
    #[must_use]
    pub fn storage_contract_fingerprint(&self) -> &str {
        &self.storage_contract_fingerprint
    }

    /// Returns the number of tensors declared in the block header.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

#[derive(Clone, Copy)]
enum PersistentPromptCacheFileKind {
    SequenceStateBlock,
    BoundaryStateSnapshot,
}

struct RequiredMetadata {
    format_version: String,
    model_id: String,
    model_revision: String,
    storage_contract_fingerprint: String,
    block_token_count: usize,
}

fn extract_required_metadata(
    metadata: &HashMap<String, String>,
    persistent_prompt_cache_block_path: &Path,
) -> Result<RequiredMetadata, PersistentPromptCacheBlockError> {
    let format_version = metadata
        .get("format_version")
        .ok_or_else(|| PersistentPromptCacheBlockError::MissingMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "format_version",
        })?
        .clone();
    let model_id = metadata
        .get("model_id")
        .ok_or_else(|| PersistentPromptCacheBlockError::MissingMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "model_id",
        })?
        .clone();
    let model_revision = metadata
        .get("model_revision")
        .ok_or_else(|| PersistentPromptCacheBlockError::MissingMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "model_revision",
        })?
        .clone();
    let storage_contract_fingerprint = metadata
        .get("storage_contract_fingerprint")
        .ok_or_else(|| PersistentPromptCacheBlockError::MissingMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "storage_contract_fingerprint",
        })?
        .clone();
    let block_token_count_text = metadata.get("block_token_count").ok_or_else(|| {
        PersistentPromptCacheBlockError::MissingMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "block_token_count",
        }
    })?;
    let block_token_count = block_token_count_text.parse::<usize>().map_err(|source| {
        PersistentPromptCacheBlockError::InvalidMetadata {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            field_name: "block_token_count",
            source,
        }
    })?;
    Ok(RequiredMetadata {
        format_version,
        model_id,
        model_revision,
        storage_contract_fingerprint,
        block_token_count,
    })
}

fn validate_metadata(
    metadata: &RequiredMetadata,
    persistent_prompt_cache_block_path: &Path,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<(), PersistentPromptCacheBlockError> {
    // A content hash proves prompt ancestry, not tensor compatibility. These
    // checks bind persisted state to its serialization contract and model.
    if metadata.format_version != PERSISTENT_PROMPT_CACHE_FORMAT_VERSION {
        return Err(PersistentPromptCacheBlockError::UnsupportedFormatVersion {
            actual_format_version: metadata.format_version.clone(),
            expected_format_version: PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.to_owned(),
        });
    }
    if metadata.model_id != persistent_prompt_cache_model_contract.model_id() {
        return Err(PersistentPromptCacheBlockError::ForeignModel {
            actual_model_id: metadata.model_id.clone(),
        });
    }
    if metadata.model_revision != persistent_prompt_cache_model_contract.model_revision() {
        return Err(PersistentPromptCacheBlockError::ForeignModelRevision {
            actual_model_revision: metadata.model_revision.clone(),
        });
    }
    if metadata.block_token_count != persistent_prompt_cache_model_contract.block_token_count() {
        return Err(PersistentPromptCacheBlockError::BlockTokenCountMismatch {
            actual_block_token_count: metadata.block_token_count,
            expected_block_token_count: persistent_prompt_cache_model_contract.block_token_count(),
        });
    }
    if metadata.storage_contract_fingerprint
        != persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex()
    {
        return Err(
            PersistentPromptCacheBlockError::InvalidModelSpecificArtifact {
                persistent_prompt_cache_block_path: persistent_prompt_cache_block_path
                    .to_path_buf(),
                description: "persistent model-state storage contract fingerprint does not match"
                    .to_owned(),
            },
        );
    }
    Ok(())
}

fn validate_tensor_layout(
    tensors: &HashMap<String, SafetensorsTensorView>,
    block_token_count: usize,
    persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
    persistent_prompt_cache_block_path: &Path,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<(), PersistentPromptCacheBlockError> {
    // Each file kind has a closed layout contract. Be deliberately strict: a
    // plausible-looking subset could produce silent wrong generation rather
    // than a recoverable cold-prefill miss.
    let expected_tensor_layouts = match persistent_prompt_cache_file_kind {
        PersistentPromptCacheFileKind::SequenceStateBlock => persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts(),
        PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
            persistent_prompt_cache_model_contract
                .decoder_cache_layout()
                .boundary_tensor_layouts()
        }
    };
    let expected_tensor_count = expected_tensor_layouts.len();
    for expected_tensor_layout in expected_tensor_layouts {
        validate_expected_tensor_layout(
            tensors,
            block_token_count,
            &expected_tensor_layout,
            persistent_prompt_cache_block_path,
        )?;
    }
    if tensors.len() != expected_tensor_count {
        return Err(PersistentPromptCacheBlockError::UnexpectedTensorCount {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            expected_tensor_count,
            actual_tensor_count: tensors.len(),
        });
    }
    Ok(())
}

fn validate_expected_tensor_layout(
    tensors: &HashMap<String, SafetensorsTensorView>,
    block_token_count: usize,
    expected_tensor_layout: &DecoderCachePersistedTensorLayout,
    persistent_prompt_cache_block_path: &Path,
) -> Result<(), PersistentPromptCacheBlockError> {
    let tensor_name = expected_tensor_layout.persistent_tensor_name();
    let tensor_view = tensors.get(&tensor_name).ok_or_else(|| {
        PersistentPromptCacheBlockError::MissingTensor {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            tensor_name: tensor_name.clone(),
        }
    })?;
    let expected_dtype = expected_tensor_layout
        .tensor_layout()
        .dtype()
        .safetensors_dtype_name();
    if tensor_view.dtype != expected_dtype {
        return Err(PersistentPromptCacheBlockError::TensorDtypeMismatch {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            tensor_name,
            expected_dtype,
            actual_dtype: tensor_view.dtype.clone(),
        });
    }
    let expected_shape = expected_tensor_layout
        .tensor_layout()
        .dimensions()
        .iter()
        .enumerate()
        .map(|(dimension_index, dimension)| {
            if Some(dimension_index) == expected_tensor_layout.tensor_layout().sequence_axis() {
                block_token_count
            } else {
                *dimension
            }
        })
        .collect::<Vec<_>>();
    if tensor_view.shape != expected_shape {
        return Err(PersistentPromptCacheBlockError::TensorShapeMismatch {
            persistent_prompt_cache_block_path: persistent_prompt_cache_block_path.to_path_buf(),
            tensor_name,
            expected_shape,
            actual_shape: tensor_view.shape.clone(),
        });
    }
    Ok(())
}

fn validate_tensor_offsets(
    tensors: &HashMap<String, SafetensorsTensorView>,
    data_section_start: u64,
    file_size_bytes: u64,
    persistent_prompt_cache_block_path: &Path,
) -> Result<(), PersistentPromptCacheBlockError> {
    // Header shape validation alone is insufficient: data offsets are also
    // untrusted and must remain inside the real file before MLX opens payloads.
    for (tensor_name, tensor_view) in tensors {
        let start_offset = tensor_view.data_offsets[0];
        let end_offset = tensor_view.data_offsets[1];
        if start_offset > end_offset {
            return Err(PersistentPromptCacheBlockError::InvalidDataOffsets {
                persistent_prompt_cache_block_path: persistent_prompt_cache_block_path
                    .to_path_buf(),
                tensor_name: tensor_name.clone(),
                start_offset,
                end_offset,
            });
        }
        let scalar_byte_count = match tensors
            .get(tensor_name)
            .map(|tensor_view| tensor_view.dtype.as_str())
        {
            Some("F16") | Some("BF16") => 2_u64,
            Some("F32") => 4_u64,
            Some(_) | None => 0_u64,
        };
        let expected_payload_byte_count = tensors
            .get(tensor_name)
            .map(|tensor_view| {
                tensor_view.shape.iter().try_fold(
                    scalar_byte_count,
                    |payload_byte_count, tensor_dimension| {
                        payload_byte_count.checked_mul(*tensor_dimension as u64)
                    },
                )
            })
            .flatten();
        if let Some(expected_payload_byte_count) = expected_payload_byte_count
            && end_offset.saturating_sub(start_offset) != expected_payload_byte_count
        {
            return Err(PersistentPromptCacheBlockError::TensorPayloadSizeMismatch {
                persistent_prompt_cache_block_path: persistent_prompt_cache_block_path
                    .to_path_buf(),
                tensor_name: tensor_name.clone(),
                expected_payload_byte_count,
                actual_payload_byte_count: end_offset.saturating_sub(start_offset),
            });
        }
        let absolute_end_offset = data_section_start.checked_add(end_offset).ok_or_else(|| {
            PersistentPromptCacheBlockError::OffsetBeyondFile {
                persistent_prompt_cache_block_path: persistent_prompt_cache_block_path
                    .to_path_buf(),
                tensor_name: tensor_name.clone(),
                end_offset,
                file_size_bytes,
            }
        })?;
        if absolute_end_offset > file_size_bytes {
            return Err(PersistentPromptCacheBlockError::OffsetBeyondFile {
                persistent_prompt_cache_block_path: persistent_prompt_cache_block_path
                    .to_path_buf(),
                tensor_name: tensor_name.clone(),
                end_offset,
                file_size_bytes,
            });
        }
    }
    Ok(())
}

fn persistent_safetensors_header_error_to_block_error(
    persistent_safetensors_header_error: PersistentSafetensorsHeaderError,
) -> PersistentPromptCacheBlockError {
    match persistent_safetensors_header_error {
        PersistentSafetensorsHeaderError::ReadFileMetadata { file_path, source } => {
            PersistentPromptCacheBlockError::ReadFileMetadata {
                persistent_prompt_cache_block_path: file_path,
                source,
            }
        }
        PersistentSafetensorsHeaderError::ReadHeaderBytes { file_path, source } => {
            PersistentPromptCacheBlockError::ReadHeaderBytes {
                persistent_prompt_cache_block_path: file_path,
                source,
            }
        }
        PersistentSafetensorsHeaderError::HeaderLengthTooLarge {
            file_path,
            header_length_bytes,
            maximum_header_length_bytes,
        } => PersistentPromptCacheBlockError::HeaderLengthTooLarge {
            persistent_prompt_cache_block_path: file_path,
            header_length_bytes,
            maximum_header_length_bytes,
        },
        PersistentSafetensorsHeaderError::TruncatedFile {
            file_path,
            expected_minimum_bytes,
            actual_file_size_bytes,
        } => PersistentPromptCacheBlockError::TruncatedFile {
            persistent_prompt_cache_block_path: file_path,
            expected_minimum_bytes,
            actual_file_size_bytes,
        },
        PersistentSafetensorsHeaderError::InvalidHeaderJson { file_path, source } => {
            PersistentPromptCacheBlockError::InvalidHeaderJson {
                persistent_prompt_cache_block_path: file_path,
                source,
            }
        }
    }
}
