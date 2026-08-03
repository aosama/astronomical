//! SSD-backed persistent prompt cache for the validated Qwen3.5-MoE artifact.
//!
//! Version 6 stores sliceable full-attention key/value blocks separately from
//! fixed-size GatedDeltaNet recurrent snapshots and rejects state from older
//! execution math. The split avoids writing the large recurrent state into
//! every KV block.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::block_key::PersistentPromptCacheBlockKey;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, expected_tensor_names, open_without_following_symlinks,
    remove_cache_owned_file_or_confirm_absent, validate_current_file_header,
};
use super::disk_store_global_quota::prepare_prompt_cache_directory_tree;
use super::disk_store_index::PersistentPromptCacheDiskStoreIndex;
use super::disk_store_scan::scan_current_format_directory;
use super::model_contract::PersistentPromptCacheModelContract;
use astronomical_runtime_integration::{MlxArray, MlxRuntime};
use std::collections::HashMap;

const KV_BLOCKS_DIRECTORY_NAME: &str = "kv_blocks";
const RECURRENT_SNAPSHOTS_DIRECTORY_NAME: &str = "recurrent_snapshots";
const VISUAL_EMBEDDINGS_DIRECTORY_NAME: &str = "visual_embeddings";

/// Valid enabled prompt-cache filesystem state for one active model under one global quota.
#[derive(Clone, Debug)]
pub struct PersistentPromptCacheDiskStoreConfig {
    active_model_prompt_cache_directory: PathBuf,
    pub(super) global_prompt_cache_root_directory: PathBuf,
    global_prompt_cache_maximum_size_bytes: u64,
    ssd_write_rate_megabytes_per_second: Option<u64>,
}

impl PersistentPromptCacheDiskStoreConfig {
    #[must_use]
    pub fn new(
        active_model_prompt_cache_directory: PathBuf,
        global_prompt_cache_root_directory: PathBuf,
        global_prompt_cache_maximum_size_bytes: u64,
    ) -> Self {
        Self::new_with_ssd_write_rate(
            active_model_prompt_cache_directory,
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
            None,
        )
    }

    #[must_use]
    pub fn new_with_ssd_write_rate(
        active_model_prompt_cache_directory: PathBuf,
        global_prompt_cache_root_directory: PathBuf,
        global_prompt_cache_maximum_size_bytes: u64,
        ssd_write_rate_megabytes_per_second: Option<u64>,
    ) -> Self {
        Self {
            active_model_prompt_cache_directory,
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
            ssd_write_rate_megabytes_per_second,
        }
    }

    #[must_use]
    pub const fn global_prompt_cache_maximum_size_bytes(&self) -> u64 {
        self.global_prompt_cache_maximum_size_bytes
    }

    #[must_use]
    pub const fn ssd_write_rate_megabytes_per_second(&self) -> Option<u64> {
        self.ssd_write_rate_megabytes_per_second
    }
}

/// Persistent, descriptor-backed SSD cache for Qwen3.5-MoE prompt-cache files.
pub struct PersistentPromptCacheDiskStore {
    pub(super) kv_blocks_directory: PathBuf,
    pub(super) recurrent_snapshots_directory: PathBuf,
    pub(crate) visual_embeddings_directory: PathBuf,
    pub(crate) global_prompt_cache_root_directory: PathBuf,
    pub(crate) global_prompt_cache_maximum_size_bytes: u64,
    pub(crate) global_prompt_cache_total_size_bytes: AtomicU64,
    pub(crate) global_visual_embedding_total_size_bytes: AtomicU64,
    pub(crate) model_contract: PersistentPromptCacheModelContract,
    tracked_files: Mutex<PersistentPromptCacheDiskStoreIndex>,
    write_operations: Mutex<()>,
}

impl PersistentPromptCacheDiskStore {
    /// Opens (or creates) the prompt-cache directory and scans current-format files.
    pub fn open(
        disk_store_config: PersistentPromptCacheDiskStoreConfig,
        model_contract: PersistentPromptCacheModelContract,
    ) -> Result<Self, PersistentPromptCacheDiskStoreError> {
        let persistent_prompt_cache_directory =
            disk_store_config.active_model_prompt_cache_directory;
        let global_prompt_cache_root_directory =
            disk_store_config.global_prompt_cache_root_directory;
        let global_prompt_cache_maximum_size_bytes =
            disk_store_config.global_prompt_cache_maximum_size_bytes;
        let kv_blocks_directory = persistent_prompt_cache_directory.join(KV_BLOCKS_DIRECTORY_NAME);
        let recurrent_snapshots_directory =
            persistent_prompt_cache_directory.join(RECURRENT_SNAPSHOTS_DIRECTORY_NAME);
        let visual_embeddings_directory =
            persistent_prompt_cache_directory.join(VISUAL_EMBEDDINGS_DIRECTORY_NAME);
        prepare_prompt_cache_directory_tree(
            &global_prompt_cache_root_directory,
            &persistent_prompt_cache_directory,
            &[
                &kv_blocks_directory,
                &recurrent_snapshots_directory,
                &visual_embeddings_directory,
            ],
        )?;
        let mut tracked_files = PersistentPromptCacheDiskStoreIndex::default();
        scan_current_format_directory(
            &kv_blocks_directory,
            PersistentPromptCacheFileKind::SequenceStateBlock,
            &mut tracked_files,
            |file, file_path| {
                validate_current_file_header(
                    PersistentPromptCacheFileKind::SequenceStateBlock,
                    file,
                    file_path,
                    &model_contract,
                )
                .is_ok()
            },
        )?;
        scan_current_format_directory(
            &recurrent_snapshots_directory,
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            &mut tracked_files,
            |file, file_path| {
                validate_current_file_header(
                    PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                    file,
                    file_path,
                    &model_contract,
                )
                .is_ok()
            },
        )?;
        let disk_store = Self {
            kv_blocks_directory,
            recurrent_snapshots_directory,
            visual_embeddings_directory,
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
            global_prompt_cache_total_size_bytes: AtomicU64::new(0),
            global_visual_embedding_total_size_bytes: AtomicU64::new(0),
            model_contract,
            tracked_files: Mutex::new(tracked_files),
            write_operations: Mutex::new(()),
        };
        disk_store.enforce_global_prompt_cache_quota()?;
        Ok(disk_store)
    }

    pub fn model_contract_ref(&self) -> &PersistentPromptCacheModelContract {
        &self.model_contract
    }

    pub fn sequence_state_block_count(&self) -> usize {
        self.lock_tracked_files().sequence_state_block_count()
    }

    pub fn boundary_state_snapshot_count(&self) -> usize {
        self.lock_tracked_files().boundary_state_snapshot_count()
    }

    pub fn visual_embedding_count(&self) -> usize {
        self.lock_tracked_files().visual_embedding_count()
    }

    pub fn has_kv_block(&self, block_hash: &[u8; 32]) -> bool {
        self.lock_tracked_files().has_kv_block(block_hash)
    }

    pub fn has_recurrent_snapshot(&self, block_hash: &[u8; 32]) -> bool {
        self.lock_tracked_files().has_recurrent_snapshot(block_hash)
    }

    pub fn has_visual_embedding(&self, visual_embedding_hash: &[u8; 32]) -> bool {
        self.lock_tracked_files()
            .has_visual_embedding(visual_embedding_hash)
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.global_prompt_cache_total_size_bytes
            .load(Ordering::Relaxed)
    }

    pub fn visual_embedding_total_size_bytes(&self) -> u64 {
        self.global_visual_embedding_total_size_bytes
            .load(Ordering::Relaxed)
    }

    pub(crate) fn lock_tracked_files(&self) -> MutexGuard<'_, PersistentPromptCacheDiskStoreIndex> {
        self.tracked_files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn lock_write_operations(&self) -> MutexGuard<'_, ()> {
        self.write_operations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn load_kv_block(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        self.load_file_kind(
            runtime,
            persistent_prompt_cache_block_key,
            PersistentPromptCacheFileKind::SequenceStateBlock,
        )
    }

    pub fn load_recurrent_snapshot(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        self.load_file_kind(
            runtime,
            persistent_prompt_cache_block_key,
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
        )
    }

    fn load_file_kind(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        file_kind: PersistentPromptCacheFileKind,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        let block_file_path = {
            let tracked_files = self.lock_tracked_files();
            let tracked_file =
                tracked_files.file(file_kind, &persistent_prompt_cache_block_key.block_hash());
            let Some(tracked_file) = tracked_file else {
                return Ok(None);
            };
            tracked_file.file_path.clone()
        };
        let block_file = match open_without_following_symlinks(&block_file_path) {
            Ok(file) => file,
            Err(open_error) => {
                if open_error.kind() == std::io::ErrorKind::NotFound {
                    self.untrack_file_and_subtract_global_accounting(
                        file_kind,
                        persistent_prompt_cache_block_key.block_hash(),
                    );
                }
                return Err(PersistentPromptCacheDiskStoreError::OpenBlockFile {
                    block_file_path,
                    source: open_error,
                });
            }
        };
        if let Err(validation_error) = validate_current_file_header(
            file_kind,
            &block_file,
            &block_file_path,
            &self.model_contract,
        ) {
            // Corrupt block: delete first (NotFound counts as already absent),
            // then untrack only after successful deletion. On any other
            // deletion failure, keep tracking and surface RemovePromptCacheFile.
            remove_cache_owned_file_or_confirm_absent(&block_file_path)?;
            self.untrack_file_and_subtract_global_accounting(
                file_kind,
                persistent_prompt_cache_block_key.block_hash(),
            );
            return Err(PersistentPromptCacheDiskStoreError::ValidateBlock {
                block_file_path,
                source: validation_error,
            });
        }
        let loaded_safetensors = runtime
            .load_safetensors(block_file)
            .map_err(|source| PersistentPromptCacheDiskStoreError::LoadSafetensors { source })?;
        let mut tensors = HashMap::new();
        for tensor_name in expected_tensor_names(file_kind, &self.model_contract) {
            let tensor = loaded_safetensors.tensor(&tensor_name).map_err(|source| {
                PersistentPromptCacheDiskStoreError::LoadSafetensors { source }
            })?;
            tensors.insert(tensor_name, tensor);
        }
        Ok(Some(tensors))
    }
}
