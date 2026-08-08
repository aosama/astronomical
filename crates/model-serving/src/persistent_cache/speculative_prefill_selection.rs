#[cfg(feature = "direct-mlx")]
use std::fs::File;
#[cfg(feature = "direct-mlx")]
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
#[cfg(feature = "direct-mlx")]
use thiserror::Error;

use super::speculative_prefill_policy::PersistentSpeculativePrefillPolicyIdentity;
#[cfg(feature = "direct-mlx")]
pub(crate) use super::speculative_prefill_selection_metadata::{
    hex_encode, selection_file_metadata_entries, selection_prompt_metadata_entries,
};

#[cfg(feature = "direct-mlx")]
use crate::safetensors::SafetensorsTensorView;

#[cfg(feature = "direct-mlx")]
use super::model_contract::PersistentPromptCacheModelContract;
#[cfg(feature = "direct-mlx")]
use super::persistent_safetensors_header::{
    PersistentSafetensorsHeader, PersistentSafetensorsHeaderError,
    read_persistent_safetensors_header,
};

/// Current on-disk format for one persisted SpecPrefill selection.
pub const PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION: &str = "2";

#[cfg(feature = "direct-mlx")]
const PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME: &str = "selected_token_positions";

/// Exact drafter and selection configuration bound to one persisted selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentSpeculativePrefillSelectionContract {
    pub(super) target_model_id: String,
    pub(super) target_model_revision: String,
    pub(super) draft_model_id: String,
    pub(super) draft_model_revision: String,
    pub(super) token_identifier_mapping_digest: [u8; 32],
    pub(super) keep_percentage: u32,
    pub(super) selection_chunck_token_count: u32,
    pub(super) mandatory_trailing_token_count: u32,
    pub(super) lookahead_token_count: u32,
    pub(super) importance_pooling_kernel_token_count: u32,
    pub(super) position_tokens: u32,
    pub(super) prompt_token_count: u32,
}

impl PersistentSpeculativePrefillSelectionContract {
    /// Binds selection metadata to the exact drafter and prompt-scoring policy.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_model_id: String,
        target_model_revision: String,
        draft_model_id: String,
        draft_model_revision: String,
        token_identifier_mapping_digest: [u8; 32],
        keep_percentage: u32,
        selection_chunck_token_count: u32,
        mandatory_trailing_token_count: u32,
        lookahead_token_count: u32,
        importance_pooling_kernel_token_count: u32,
        position_tokens: u32,
        prompt_token_count: u32,
    ) -> Self {
        Self {
            target_model_id,
            target_model_revision,
            draft_model_id,
            draft_model_revision,
            token_identifier_mapping_digest,
            keep_percentage,
            selection_chunck_token_count,
            mandatory_trailing_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
            position_tokens,
            prompt_token_count,
        }
    }

    /// Returns the target, drafter, and keep percentage bound to this selection.
    #[must_use]
    pub fn policy_identity(&self) -> PersistentSpeculativePrefillPolicyIdentity {
        PersistentSpeculativePrefillPolicyIdentity::new(
            self.target_model_id.clone(),
            self.target_model_revision.clone(),
            self.draft_model_id.clone(),
            self.draft_model_revision.clone(),
            self.keep_percentage,
        )
    }

    /// Returns the model identity used by the drafter selection.
    #[must_use]
    pub fn draft_model_id(&self) -> &str {
        &self.draft_model_id
    }

    /// Returns the exact drafter artifact revision.
    #[must_use]
    pub fn draft_model_revision(&self) -> &str {
        &self.draft_model_revision
    }

    /// Returns the target/drafter tokenizer mapping digest.
    #[must_use]
    pub const fn token_identifier_mapping_digest(&self) -> &[u8; 32] {
        &self.token_identifier_mapping_digest
    }

    /// Returns the number of prompt positions represented by the selection.
    #[must_use]
    pub const fn prompt_token_count(&self) -> u32 {
        self.prompt_token_count
    }

    /// Hashes the full selection identity, including the complete prompt token sequence.
    #[must_use]
    pub fn selection_identity_hash(&self, prompt_token_ids: &[u32]) -> [u8; 32] {
        let mut selection_identity_hasher = Sha256::new();
        append_length_prefixed_bytes(
            &mut selection_identity_hasher,
            self.target_model_id.as_bytes(),
        );
        append_length_prefixed_bytes(
            &mut selection_identity_hasher,
            self.target_model_revision.as_bytes(),
        );
        append_length_prefixed_bytes(
            &mut selection_identity_hasher,
            self.draft_model_id.as_bytes(),
        );
        append_length_prefixed_bytes(
            &mut selection_identity_hasher,
            self.draft_model_revision.as_bytes(),
        );
        selection_identity_hasher.update(self.token_identifier_mapping_digest);
        for selection_configuration_number in [
            self.keep_percentage,
            self.selection_chunck_token_count,
            self.mandatory_trailing_token_count,
            self.lookahead_token_count,
            self.importance_pooling_kernel_token_count,
            self.position_tokens,
            self.prompt_token_count,
        ] {
            selection_identity_hasher.update(selection_configuration_number.to_le_bytes());
        }
        for prompt_token_id in prompt_token_ids {
            selection_identity_hasher.update(prompt_token_id.to_le_bytes());
        }
        selection_identity_hasher.finalize().into()
    }
}

/// Bounded metadata parsed from a persisted selection file header.
#[cfg(feature = "direct-mlx")]
#[derive(Debug)]
pub(crate) struct PersistentSpeculativePrefillSelectionFileHeader {
    selected_token_position_count: usize,
    policy_identity: PersistentSpeculativePrefillPolicyIdentity,
}

#[cfg(feature = "direct-mlx")]
impl PersistentSpeculativePrefillSelectionFileHeader {
    pub(crate) fn read_model_bound_from_file(
        selection_file: &File,
        selection_file_path: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentSpeculativePrefillSelectionFileError> {
        let parsed_header = read_selection_header(selection_file, selection_file_path)?;
        validate_model_bound_metadata(
            &parsed_header,
            selection_file_path,
            persistent_prompt_cache_model_contract,
        )?;
        Ok(Self {
            selected_token_position_count: selected_token_position_count(
                &parsed_header,
                selection_file_path,
            )?,
            policy_identity: selection_policy_identity(&parsed_header, selection_file_path)?,
        })
    }

    pub(crate) fn read_for_contract_from_file(
        selection_file: &File,
        selection_file_path: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
        selection_contract: &PersistentSpeculativePrefillSelectionContract,
        expected_selection_identity_hash: [u8; 32],
    ) -> Result<Self, PersistentSpeculativePrefillSelectionFileError> {
        let parsed_header = read_selection_header(selection_file, selection_file_path)?;
        validate_model_bound_metadata(
            &parsed_header,
            selection_file_path,
            persistent_prompt_cache_model_contract,
        )?;
        validate_contract_metadata(
            &parsed_header,
            selection_file_path,
            selection_contract,
            expected_selection_identity_hash,
        )?;
        Ok(Self {
            selected_token_position_count: selected_token_position_count(
                &parsed_header,
                selection_file_path,
            )?,
            policy_identity: selection_policy_identity(&parsed_header, selection_file_path)?,
        })
    }

    pub(crate) const fn selected_token_position_count(&self) -> usize {
        self.selected_token_position_count
    }

    pub(crate) const fn policy_identity(&self) -> &PersistentSpeculativePrefillPolicyIdentity {
        &self.policy_identity
    }
}

/// One malformed or incompatible persisted SpecPrefill selection file.
#[cfg(feature = "direct-mlx")]
#[derive(Debug, Error)]
pub(crate) enum PersistentSpeculativePrefillSelectionFileError {
    #[error(
        "failed to read persisted speculative-prefill selection header at {selection_file_path:?}"
    )]
    ReadHeader {
        selection_file_path: PathBuf,
        #[source]
        source: PersistentSafetensorsHeaderError,
    },
    #[error(
        "persisted speculative-prefill selection at {selection_file_path:?} is missing metadata '{metadata_name}'"
    )]
    MissingMetadata {
        selection_file_path: PathBuf,
        metadata_name: &'static str,
    },
    #[error(
        "persisted speculative-prefill selection at {selection_file_path:?} has metadata '{metadata_name}'={actual_metadata_value:?}, expected {expected_metadata_value:?}"
    )]
    MetadataMismatch {
        selection_file_path: PathBuf,
        metadata_name: &'static str,
        actual_metadata_value: String,
        expected_metadata_value: String,
    },
    #[error(
        "persisted speculative-prefill selection at {selection_file_path:?} has an invalid tensor layout"
    )]
    InvalidTensorLayout { selection_file_path: PathBuf },
    #[error(
        "persisted speculative-prefill selection at {selection_file_path:?} has invalid metadata '{metadata_name}'"
    )]
    InvalidMetadata {
        selection_file_path: PathBuf,
        metadata_name: &'static str,
    },
    #[error(
        "persisted speculative-prefill selection at {selection_file_path:?} has tensor data outside the file"
    )]
    TensorDataOutsideFile { selection_file_path: PathBuf },
}

#[cfg(feature = "direct-mlx")]
fn selection_policy_identity(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<PersistentSpeculativePrefillPolicyIdentity, PersistentSpeculativePrefillSelectionFileError>
{
    let required_metadata = |metadata_name: &'static str| {
        parsed_header.metadata.get(metadata_name).cloned().ok_or_else(|| {
            PersistentSpeculativePrefillSelectionFileError::MissingMetadata {
                selection_file_path: selection_file_path.to_path_buf(),
                metadata_name,
            }
        })
    };
    let keep_percentage = required_metadata("keep_percentage")?
        .parse::<u32>()
        .map_err(|_| PersistentSpeculativePrefillSelectionFileError::InvalidMetadata {
            selection_file_path: selection_file_path.to_path_buf(),
            metadata_name: "keep_percentage",
        })?;
    Ok(PersistentSpeculativePrefillPolicyIdentity::new(
        required_metadata("target_model_id")?,
        required_metadata("target_model_revision")?,
        required_metadata("model_id")?,
        required_metadata("model_revision")?,
        keep_percentage,
    ))
}

#[cfg(feature = "direct-mlx")]
fn read_selection_header(
    selection_file: &File,
    selection_file_path: &Path,
) -> Result<PersistentSafetensorsHeader, PersistentSpeculativePrefillSelectionFileError> {
    read_persistent_safetensors_header(selection_file, selection_file_path).map_err(|source| {
        PersistentSpeculativePrefillSelectionFileError::ReadHeader {
            selection_file_path: selection_file_path.to_path_buf(),
            source,
        }
    })
}

#[cfg(feature = "direct-mlx")]
fn validate_model_bound_metadata(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    validate_metadata_text(
        parsed_header,
        selection_file_path,
        "format_version",
        PERSISTENT_SPECULATIVE_PREFILL_SELECTION_FORMAT_VERSION,
    )?;
    validate_metadata_text(
        parsed_header,
        selection_file_path,
        "model_id",
        persistent_prompt_cache_model_contract.model_id(),
    )?;
    validate_metadata_text(
        parsed_header,
        selection_file_path,
        "model_revision",
        persistent_prompt_cache_model_contract.model_revision(),
    )?;
    validate_selection_tensor_layout(parsed_header, selection_file_path)
}

#[cfg(feature = "direct-mlx")]
fn validate_contract_metadata(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
    selection_contract: &PersistentSpeculativePrefillSelectionContract,
    expected_selection_identity_hash: [u8; 32],
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    let expected_metadata_entries = [
        (
            "target_model_id",
            selection_contract.target_model_id.clone(),
        ),
        (
            "target_model_revision",
            selection_contract.target_model_revision.clone(),
        ),
        (
            "token_identifier_mapping_digest",
            hex_encode(*selection_contract.token_identifier_mapping_digest()),
        ),
        (
            "keep_percentage",
            selection_contract.keep_percentage.to_string(),
        ),
        (
            "selection_chunck_token_count",
            selection_contract.selection_chunck_token_count.to_string(),
        ),
        (
            "mandatory_trailing_token_count",
            selection_contract
                .mandatory_trailing_token_count
                .to_string(),
        ),
        (
            "lookahead_token_count",
            selection_contract.lookahead_token_count.to_string(),
        ),
        (
            "importance_pooling_kernel_token_count",
            selection_contract
                .importance_pooling_kernel_token_count
                .to_string(),
        ),
        (
            "position_tokens",
            selection_contract.position_tokens.to_string(),
        ),
        (
            "prompt_token_count",
            selection_contract.prompt_token_count.to_string(),
        ),
        (
            "selection_identity",
            hex_encode(expected_selection_identity_hash),
        ),
    ];
    for (metadata_name, expected_metadata_value) in expected_metadata_entries {
        validate_metadata_text(
            parsed_header,
            selection_file_path,
            metadata_name,
            expected_metadata_value.as_str(),
        )?;
    }
    Ok(())
}

#[cfg(feature = "direct-mlx")]
fn validate_metadata_text(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
    metadata_name: &'static str,
    expected_metadata_value: &str,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    let actual_metadata_value = parsed_header.metadata.get(metadata_name).ok_or_else(|| {
        PersistentSpeculativePrefillSelectionFileError::MissingMetadata {
            selection_file_path: selection_file_path.to_path_buf(),
            metadata_name,
        }
    })?;
    if actual_metadata_value != expected_metadata_value {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::MetadataMismatch {
                selection_file_path: selection_file_path.to_path_buf(),
                metadata_name,
                actual_metadata_value: actual_metadata_value.clone(),
                expected_metadata_value: expected_metadata_value.to_owned(),
            },
        );
    }
    Ok(())
}

#[cfg(feature = "direct-mlx")]
fn validate_selection_tensor_layout(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    if parsed_header.tensor_views.len() != 1 {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::InvalidTensorLayout {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        );
    }
    let Some(selected_token_positions_tensor) = parsed_header
        .tensor_views
        .get(PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME)
    else {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::InvalidTensorLayout {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        );
    };
    if selected_token_positions_tensor.dtype != "U32"
        || selected_token_positions_tensor.shape.len() != 1
        || selected_token_positions_tensor.shape[0] == 0
    {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::InvalidTensorLayout {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        );
    }
    validate_tensor_offsets(
        selected_token_positions_tensor,
        parsed_header,
        selection_file_path,
    )
}

#[cfg(feature = "direct-mlx")]
fn selected_token_position_count(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<usize, PersistentSpeculativePrefillSelectionFileError> {
    parsed_header
        .tensor_views
        .get(PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME)
        .map(|selected_token_positions_tensor| selected_token_positions_tensor.shape[0])
        .ok_or_else(
            || PersistentSpeculativePrefillSelectionFileError::InvalidTensorLayout {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        )
}

#[cfg(feature = "direct-mlx")]
fn validate_tensor_offsets(
    selected_token_positions_tensor: &SafetensorsTensorView,
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    let [data_start_offset, data_end_offset] = selected_token_positions_tensor.data_offsets;
    let Some(absolute_data_end_offset) = parsed_header
        .data_section_start_bytes
        .checked_add(data_end_offset)
    else {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::TensorDataOutsideFile {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        );
    };
    if data_start_offset > data_end_offset
        || absolute_data_end_offset > parsed_header.file_size_bytes
    {
        return Err(
            PersistentSpeculativePrefillSelectionFileError::TensorDataOutsideFile {
                selection_file_path: selection_file_path.to_path_buf(),
            },
        );
    }
    Ok(())
}

fn append_length_prefixed_bytes(selection_identity_hasher: &mut Sha256, bytes: &[u8]) {
    selection_identity_hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    selection_identity_hasher.update(bytes);
}
