//! Durable identity and topology record for one committed prompt-cache block.
//!
//! The directory name proves only the block's content hash. This manifest also
//! binds that hash to its ordinal position, parent, model storage geometry, and
//! required state-file kinds. Readers must validate all of those fields before
//! treating files in the directory as one link in a restorable chain.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::block_format::PERSISTENT_PROMPT_CACHE_FORMAT_VERSION;
use super::block_key::PersistentPromptCacheBlockKey;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{hex_encode, remove_cache_owned_file_or_confirm_absent};
use super::model_contract::PersistentPromptCacheModelContract;

pub(crate) const BLOCK_MANIFEST_FILE_NAME: &str = "manifest.json";
pub(crate) const SEQUENCE_STATE_FILE_NAME: &str = "sequence.safetensors";
pub(crate) const BOUNDARY_STATE_FILE_NAME: &str = "boundary.safetensors";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct PersistentPromptCacheBlockManifest {
    // These fields intentionally duplicate facts available elsewhere. Keeping
    // the complete contract in each block makes startup validation local and
    // prevents directory layout or file presence from becoming implicit truth.
    format_version: String,
    block_hash: String,
    block_index: u32,
    parent_block_hash: Option<String>,
    storage_contract_fingerprint: String,
    has_sequence_state: bool,
    has_boundary_state: bool,
}

impl PersistentPromptCacheBlockManifest {
    pub(crate) fn new(
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        parent_persistent_prompt_cache_block_key: Option<&PersistentPromptCacheBlockKey>,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Self {
        Self {
            format_version: PERSISTENT_PROMPT_CACHE_FORMAT_VERSION.to_owned(),
            block_hash: hex_encode(persistent_prompt_cache_block_key.block_hash()),
            block_index: persistent_prompt_cache_block_key.block_index(),
            parent_block_hash: parent_persistent_prompt_cache_block_key
                .map(|parent_block_key| hex_encode(parent_block_key.block_hash())),
            storage_contract_fingerprint: persistent_prompt_cache_model_contract
                .storage_contract_fingerprint_hex(),
            has_sequence_state: persistent_prompt_cache_model_contract.has_sequence_state(),
            has_boundary_state: persistent_prompt_cache_model_contract.has_boundary_state(),
        }
    }

    pub(crate) fn read_from_block_directory(
        block_directory: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentPromptCacheDiskStoreError> {
        let block_manifest = Self::read_unvalidated_from_block_directory(block_directory)?;
        let manifest_file_path = block_directory.join(BLOCK_MANIFEST_FILE_NAME);
        block_manifest.validate(&manifest_file_path, persistent_prompt_cache_model_contract)?;
        Ok(block_manifest)
    }

    pub(crate) fn read_unvalidated_from_block_directory(
        block_directory: &Path,
    ) -> Result<Self, PersistentPromptCacheDiskStoreError> {
        let manifest_file_path = block_directory.join(BLOCK_MANIFEST_FILE_NAME);
        let manifest_text = fs::read_to_string(&manifest_file_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockManifest {
                manifest_file_path: manifest_file_path.clone(),
                source,
            }
        })?;
        let block_manifest = serde_json::from_str::<Self>(&manifest_text).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ParseBlockManifest {
                manifest_file_path: manifest_file_path.clone(),
                source,
            }
        })?;
        Ok(block_manifest)
    }

    pub(crate) fn write_to_staging_directory(
        &self,
        staging_block_directory: &Path,
    ) -> Result<PathBuf, PersistentPromptCacheDiskStoreError> {
        // The manifest is itself committed inside the private staging
        // directory. The enclosing block directory is published only after all
        // state files and this manifest are durable.
        let manifest_file_path = staging_block_directory.join(BLOCK_MANIFEST_FILE_NAME);
        let temporary_manifest_file_path = staging_block_directory.join("manifest.json.tmp");
        remove_cache_owned_file_or_confirm_absent(&temporary_manifest_file_path)?;
        let manifest_bytes = serde_json::to_vec(self).map_err(|source| {
            PersistentPromptCacheDiskStoreError::SerializeBlockManifest { source }
        })?;
        let mut temporary_manifest_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary_manifest_file_path)
            .map_err(|source| PersistentPromptCacheDiskStoreError::OpenTempFile {
                temp_file_path: temporary_manifest_file_path.clone(),
                source,
            })?;
        if let Err(source) = temporary_manifest_file.write_all(&manifest_bytes) {
            remove_cache_owned_file_or_confirm_absent(&temporary_manifest_file_path)?;
            return Err(PersistentPromptCacheDiskStoreError::WriteTempFile {
                temp_file_path: temporary_manifest_file_path,
                source,
            });
        }
        if let Err(source) = temporary_manifest_file.sync_all() {
            remove_cache_owned_file_or_confirm_absent(&temporary_manifest_file_path)?;
            return Err(PersistentPromptCacheDiskStoreError::SynchronizeTempFile {
                temp_file_path: temporary_manifest_file_path,
                source,
            });
        }
        drop(temporary_manifest_file);
        if let Err(source) = fs::rename(&temporary_manifest_file_path, &manifest_file_path) {
            let rename_error = PersistentPromptCacheDiskStoreError::RenameTempFile {
                temp_file_path: temporary_manifest_file_path.clone(),
                block_file_path: manifest_file_path.clone(),
                source,
            };
            remove_cache_owned_file_or_confirm_absent(&temporary_manifest_file_path)?;
            return Err(rename_error);
        }
        Ok(manifest_file_path)
    }

    pub(crate) fn block_hash(&self) -> Result<[u8; 32], PersistentPromptCacheDiskStoreError> {
        parse_block_hash_hex(&self.block_hash).ok_or_else(|| {
            PersistentPromptCacheDiskStoreError::InvalidBlockManifest {
                manifest_file_path: PathBuf::from(BLOCK_MANIFEST_FILE_NAME),
                description: "block hash is not a 32-byte lowercase hexadecimal value".to_owned(),
            }
        })
    }

    pub(crate) fn parent_block_hash(&self) -> Option<[u8; 32]> {
        self.parent_block_hash
            .as_ref()
            .and_then(|parent_block_hash| parse_block_hash_hex(parent_block_hash))
    }

    pub(crate) const fn block_index(&self) -> u32 {
        self.block_index
    }

    pub(crate) const fn has_sequence_state(&self) -> bool {
        self.has_sequence_state
    }

    pub(crate) const fn has_boundary_state(&self) -> bool {
        self.has_boundary_state
    }

    pub(crate) fn storage_contract_fingerprint(&self) -> &str {
        &self.storage_contract_fingerprint
    }

    fn validate(
        &self,
        manifest_file_path: &Path,
        persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        // Validation is deliberately fail-closed. A block from another format,
        // model revision, tensor layout, or state topology must never be joined
        // to the active request merely because its content hash parses.
        if self.format_version != PERSISTENT_PROMPT_CACHE_FORMAT_VERSION {
            return Err(invalid_manifest(
                manifest_file_path,
                "format version does not match the active prompt-cache format",
            ));
        }
        if self.storage_contract_fingerprint
            != persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex()
        {
            return Err(invalid_manifest(
                manifest_file_path,
                "storage contract fingerprint does not match the active model",
            ));
        }
        if self.has_sequence_state != persistent_prompt_cache_model_contract.has_sequence_state()
            || self.has_boundary_state
                != persistent_prompt_cache_model_contract.has_boundary_state()
        {
            return Err(invalid_manifest(
                manifest_file_path,
                "state topology does not match the active model contract",
            ));
        }
        if parse_block_hash_hex(&self.block_hash).is_none()
            || self
                .parent_block_hash
                .as_ref()
                .is_some_and(|parent_block_hash| parse_block_hash_hex(parent_block_hash).is_none())
        {
            return Err(invalid_manifest(
                manifest_file_path,
                "block ancestry contains an invalid hash",
            ));
        }
        Ok(())
    }
}

fn invalid_manifest(
    manifest_file_path: &Path,
    description: &'static str,
) -> PersistentPromptCacheDiskStoreError {
    PersistentPromptCacheDiskStoreError::InvalidBlockManifest {
        manifest_file_path: manifest_file_path.to_path_buf(),
        description: description.to_owned(),
    }
}

fn parse_block_hash_hex(block_hash_hex: &str) -> Option<[u8; 32]> {
    // The writer emits a canonical 64-character representation. Parsing still
    // validates exact width and every byte pair before returning binary identity.
    if block_hash_hex.len() != 64 {
        return None;
    }
    let mut block_hash = [0_u8; 32];
    for (block_hash_byte_index, block_hash_byte) in block_hash.iter_mut().enumerate() {
        let byte_hex = &block_hash_hex[block_hash_byte_index * 2..block_hash_byte_index * 2 + 2];
        *block_hash_byte = u8::from_str_radix(byte_hex, 16).ok()?;
    }
    Some(block_hash)
}
