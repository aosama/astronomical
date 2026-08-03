//! Closed, bounded validation for persisted Qwen3.5-MoE projected embeddings.

use std::fs::File;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::PersistentVisualEmbeddingModelContract;
use super::visual_embedding_key::{
    PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION, PersistentVisualEmbeddingKey,
};
use crate::persistent_cache::persistent_safetensors_header::{
    PersistentSafetensorsHeader, read_persistent_safetensors_header,
};

const VISUAL_EMBEDDING_TENSOR_NAME: &str = "visual_embeddings";
const VISUAL_EMBEDDING_BFLOAT16_BYTE_COUNT: usize = 2;

/// Validated metadata for one persisted projected visual embedding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentVisualEmbeddingFileHeader {
    format_version: String,
    model_id: String,
    model_revision: String,
    encoded_image_sha256: [u8; 32],
    visual_token_count: usize,
}

impl PersistentVisualEmbeddingFileHeader {
    /// Reads and validates one visual embedding header without loading its payload.
    pub fn read_from_file(
        visual_embedding_file: &File,
        visual_embedding_file_path: &Path,
        persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
    ) -> Result<Self, PersistentVisualEmbeddingFileError> {
        let parsed_safetensors_header =
            read_persistent_safetensors_header(visual_embedding_file, visual_embedding_file_path)
                .map_err(|header_error| {
                PersistentVisualEmbeddingFileError::Header(Box::new(header_error))
            })?;
        let format_version = required_metadata(
            &parsed_safetensors_header.metadata,
            "format_version",
            visual_embedding_file_path,
        )?;
        let model_id = required_metadata(
            &parsed_safetensors_header.metadata,
            "model_id",
            visual_embedding_file_path,
        )?;
        let model_revision = required_metadata(
            &parsed_safetensors_header.metadata,
            "model_revision",
            visual_embedding_file_path,
        )?;
        let encoded_image_sha256_text = required_metadata(
            &parsed_safetensors_header.metadata,
            "encoded_image_sha256",
            visual_embedding_file_path,
        )?;
        let encoded_image_sha256 =
            decode_lowercase_sha256(&encoded_image_sha256_text, visual_embedding_file_path)?;
        let visual_token_count_text = required_metadata(
            &parsed_safetensors_header.metadata,
            "visual_token_count",
            visual_embedding_file_path,
        )?;
        let visual_token_count = visual_token_count_text.parse::<usize>().map_err(|source| {
            PersistentVisualEmbeddingFileError::InvalidMetadata {
                visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
                field_name: "visual_token_count",
                source,
            }
        })?;
        validate_metadata(
            &format_version,
            &model_id,
            &model_revision,
            visual_token_count,
            visual_embedding_file_path,
            persistent_visual_embedding_model_contract,
        )?;
        validate_filename_digest(
            visual_embedding_file_path,
            encoded_image_sha256,
            persistent_visual_embedding_model_contract,
        )?;
        validate_tensor_layout(
            &parsed_safetensors_header,
            visual_token_count,
            visual_embedding_file_path,
            persistent_visual_embedding_model_contract,
        )?;
        Ok(Self {
            format_version,
            model_id,
            model_revision,
            encoded_image_sha256,
            visual_token_count,
        })
    }

    /// Returns the visual file-format version.
    #[must_use]
    pub fn format_version(&self) -> &str {
        &self.format_version
    }

    /// Returns the validated model identity stamped into the file.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Returns the validated model revision stamped into the file.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// Returns the exact encoded-image digest bound to the file.
    #[must_use]
    pub const fn encoded_image_sha256(&self) -> [u8; 32] {
        self.encoded_image_sha256
    }

    /// Returns the number of projected visual rows in the file.
    #[must_use]
    pub const fn visual_token_count(&self) -> usize {
        self.visual_token_count
    }
}

fn required_metadata(
    metadata: &std::collections::HashMap<String, String>,
    field_name: &'static str,
    visual_embedding_file_path: &Path,
) -> Result<String, PersistentVisualEmbeddingFileError> {
    metadata.get(field_name).cloned().ok_or_else(|| {
        PersistentVisualEmbeddingFileError::MissingMetadata {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            field_name,
        }
    })
}

fn validate_metadata(
    format_version: &str,
    model_id: &str,
    model_revision: &str,
    visual_token_count: usize,
    visual_embedding_file_path: &Path,
    persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
) -> Result<(), PersistentVisualEmbeddingFileError> {
    if format_version != PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION {
        return Err(
            PersistentVisualEmbeddingFileError::UnsupportedFormatVersion {
                visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
                actual_format_version: format_version.to_owned(),
                expected_format_version: PERSISTENT_VISUAL_EMBEDDING_FORMAT_VERSION,
            },
        );
    }
    if model_id != persistent_visual_embedding_model_contract.model_id() {
        return Err(PersistentVisualEmbeddingFileError::ForeignModel {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            actual_model_id: model_id.to_owned(),
        });
    }
    if model_revision != persistent_visual_embedding_model_contract.model_revision() {
        return Err(PersistentVisualEmbeddingFileError::ForeignModelRevision {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            actual_model_revision: model_revision.to_owned(),
        });
    }
    let maximum_visual_embedding_token_count =
        persistent_visual_embedding_model_contract.maximum_visual_embedding_token_count();
    if visual_token_count == 0 || visual_token_count > maximum_visual_embedding_token_count {
        return Err(
            PersistentVisualEmbeddingFileError::VisualTokenCountOutOfRange {
                visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
                actual_visual_token_count: visual_token_count,
                maximum_visual_token_count: maximum_visual_embedding_token_count,
            },
        );
    }
    Ok(())
}

fn validate_filename_digest(
    visual_embedding_file_path: &Path,
    encoded_image_sha256: [u8; 32],
    persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
) -> Result<(), PersistentVisualEmbeddingFileError> {
    let expected_visual_embedding_hash = PersistentVisualEmbeddingKey::for_image(
        encoded_image_sha256,
        persistent_visual_embedding_model_contract.model_id(),
        persistent_visual_embedding_model_contract.model_revision(),
    )
    .visual_embedding_hash();
    let actual_visual_embedding_hash = parse_digest_from_file_path(visual_embedding_file_path)
        .ok_or_else(|| PersistentVisualEmbeddingFileError::InvalidFileName {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        })?;
    if actual_visual_embedding_hash != expected_visual_embedding_hash {
        return Err(PersistentVisualEmbeddingFileError::FileNameDigestMismatch {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_tensor_layout(
    parsed_safetensors_header: &PersistentSafetensorsHeader,
    visual_token_count: usize,
    visual_embedding_file_path: &Path,
    persistent_visual_embedding_model_contract: &PersistentVisualEmbeddingModelContract,
) -> Result<(), PersistentVisualEmbeddingFileError> {
    if parsed_safetensors_header.tensor_views.len() != 1 {
        return Err(PersistentVisualEmbeddingFileError::UnexpectedTensorCount {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            actual_tensor_count: parsed_safetensors_header.tensor_views.len(),
        });
    }
    let visual_embedding_tensor = parsed_safetensors_header
        .tensor_views
        .get(VISUAL_EMBEDDING_TENSOR_NAME)
        .ok_or_else(|| PersistentVisualEmbeddingFileError::MissingTensor {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        })?;
    if visual_embedding_tensor.dtype != "BF16" {
        return Err(PersistentVisualEmbeddingFileError::TensorDtypeMismatch {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            actual_dtype: visual_embedding_tensor.dtype.clone(),
        });
    }
    let expected_tensor_shape = persistent_visual_embedding_model_contract
        .visual_embedding_shape(visual_token_count)
        .to_vec();
    if visual_embedding_tensor.shape != expected_tensor_shape {
        return Err(PersistentVisualEmbeddingFileError::TensorShapeMismatch {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            expected_shape: expected_tensor_shape,
            actual_shape: visual_embedding_tensor.shape.clone(),
        });
    }
    let expected_payload_byte_count = visual_token_count
        .checked_mul(persistent_visual_embedding_model_contract.visual_embedding_hidden_size())
        .and_then(|element_count| element_count.checked_mul(VISUAL_EMBEDDING_BFLOAT16_BYTE_COUNT))
        .ok_or_else(|| PersistentVisualEmbeddingFileError::PayloadSizeOverflow {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        })?;
    let [payload_start_bytes, payload_end_bytes] = visual_embedding_tensor.data_offsets;
    if payload_start_bytes != 0 || payload_end_bytes != expected_payload_byte_count as u64 {
        return Err(
            PersistentVisualEmbeddingFileError::InvalidTensorDataOffsets {
                visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
                payload_start_bytes,
                payload_end_bytes,
                expected_payload_byte_count,
            },
        );
    }
    let absolute_payload_end_bytes = parsed_safetensors_header
        .data_section_start_bytes
        .checked_add(payload_end_bytes)
        .ok_or_else(|| PersistentVisualEmbeddingFileError::PayloadBeyondFile {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            payload_end_bytes,
            file_size_bytes: parsed_safetensors_header.file_size_bytes,
        })?;
    if absolute_payload_end_bytes > parsed_safetensors_header.file_size_bytes {
        return Err(PersistentVisualEmbeddingFileError::PayloadBeyondFile {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
            payload_end_bytes,
            file_size_bytes: parsed_safetensors_header.file_size_bytes,
        });
    }
    Ok(())
}

fn decode_lowercase_sha256(
    digest_text: &str,
    visual_embedding_file_path: &Path,
) -> Result<[u8; 32], PersistentVisualEmbeddingFileError> {
    if digest_text.len() != 64 || digest_text != digest_text.to_ascii_lowercase() {
        return Err(PersistentVisualEmbeddingFileError::InvalidImageDigest {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        });
    }
    let mut decoded_digest = [0_u8; 32];
    for (digest_byte_index, decoded_digest_byte) in decoded_digest.iter_mut().enumerate() {
        *decoded_digest_byte = u8::from_str_radix(
            &digest_text[digest_byte_index * 2..digest_byte_index * 2 + 2],
            16,
        )
        .map_err(|_| PersistentVisualEmbeddingFileError::InvalidImageDigest {
            visual_embedding_file_path: visual_embedding_file_path.to_path_buf(),
        })?;
    }
    Ok(decoded_digest)
}

fn parse_digest_from_file_path(file_path: &Path) -> Option<[u8; 32]> {
    if file_path.extension()?.to_str()? != "safetensors" {
        return None;
    }
    let file_stem = file_path.file_stem()?.to_str()?;
    if file_stem.len() != 64 || file_stem != file_stem.to_ascii_lowercase() {
        return None;
    }
    let mut visual_embedding_hash = [0_u8; 32];
    for (hash_byte_index, visual_embedding_hash_byte) in
        visual_embedding_hash.iter_mut().enumerate()
    {
        *visual_embedding_hash_byte =
            u8::from_str_radix(&file_stem[hash_byte_index * 2..hash_byte_index * 2 + 2], 16)
                .ok()?;
    }
    Some(visual_embedding_hash)
}

/// One visual embedding file did not satisfy the expected on-disk contract.
#[derive(Debug, Error)]
pub enum PersistentVisualEmbeddingFileError {
    #[error("failed to read visual embedding header")]
    Header(#[source] Box<dyn std::error::Error + Send + Sync>),
    #[error(
        "visual embedding file {visual_embedding_file_path:?} is missing metadata {field_name}"
    )]
    MissingMetadata {
        visual_embedding_file_path: PathBuf,
        field_name: &'static str,
    },
    #[error("visual embedding metadata {field_name} in {visual_embedding_file_path:?} is invalid")]
    InvalidMetadata {
        visual_embedding_file_path: PathBuf,
        field_name: &'static str,
        #[source]
        source: std::num::ParseIntError,
    },
    #[error("visual embedding file {visual_embedding_file_path:?} has an invalid image digest")]
    InvalidImageDigest { visual_embedding_file_path: PathBuf },
    #[error(
        "visual embedding file format is {actual_format_version}, expected {expected_format_version}"
    )]
    UnsupportedFormatVersion {
        visual_embedding_file_path: PathBuf,
        actual_format_version: String,
        expected_format_version: &'static str,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} belongs to foreign model {actual_model_id}"
    )]
    ForeignModel {
        visual_embedding_file_path: PathBuf,
        actual_model_id: String,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} belongs to foreign model revision {actual_model_revision}"
    )]
    ForeignModelRevision {
        visual_embedding_file_path: PathBuf,
        actual_model_revision: String,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} has {actual_visual_token_count} rows, maximum {maximum_visual_token_count}"
    )]
    VisualTokenCountOutOfRange {
        visual_embedding_file_path: PathBuf,
        actual_visual_token_count: usize,
        maximum_visual_token_count: usize,
    },
    #[error(
        "visual embedding file name {visual_embedding_file_path:?} is not a lowercase SHA-256 identity"
    )]
    InvalidFileName { visual_embedding_file_path: PathBuf },
    #[error(
        "visual embedding file name does not match its metadata digest: {visual_embedding_file_path:?}"
    )]
    FileNameDigestMismatch { visual_embedding_file_path: PathBuf },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} has {actual_tensor_count} tensors instead of one"
    )]
    UnexpectedTensorCount {
        visual_embedding_file_path: PathBuf,
        actual_tensor_count: usize,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} is missing tensor visual_embeddings"
    )]
    MissingTensor { visual_embedding_file_path: PathBuf },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} has dtype {actual_dtype}, expected BF16"
    )]
    TensorDtypeMismatch {
        visual_embedding_file_path: PathBuf,
        actual_dtype: String,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} has shape {actual_shape:?}, expected {expected_shape:?}"
    )]
    TensorShapeMismatch {
        visual_embedding_file_path: PathBuf,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },
    #[error("visual embedding file {visual_embedding_file_path:?} payload size overflowed")]
    PayloadSizeOverflow { visual_embedding_file_path: PathBuf },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} has offsets [{payload_start_bytes}, {payload_end_bytes}], expected [0, {expected_payload_byte_count}]"
    )]
    InvalidTensorDataOffsets {
        visual_embedding_file_path: PathBuf,
        payload_start_bytes: u64,
        payload_end_bytes: u64,
        expected_payload_byte_count: usize,
    },
    #[error(
        "visual embedding file {visual_embedding_file_path:?} payload ends at {payload_end_bytes}, beyond {file_size_bytes} bytes"
    )]
    PayloadBeyondFile {
        visual_embedding_file_path: PathBuf,
        payload_end_bytes: u64,
        file_size_bytes: u64,
    },
}
