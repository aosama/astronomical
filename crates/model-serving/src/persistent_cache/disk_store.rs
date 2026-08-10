//! SSD-backed, architecture-neutral persistent decoder-state cache.
//!
//! Current-format state is published as complete atomic block directories with
//! manifest-bound ancestry and contract-derived sequence and boundary files.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::block_key::PersistentPromptCacheBlockKey;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, expected_tensor_names, open_without_following_symlinks,
    remove_cache_owned_file_or_confirm_absent, validate_current_file_header,
};
use super::disk_store_global_quota::prepare_prompt_cache_directory_tree;
use super::disk_store_index::PersistentPromptCacheDiskStoreIndex;
use super::disk_store_scan::{
    scan_current_format_block_directories, scan_current_format_directory,
};
use super::model_contract::PersistentPromptCacheModelContract;
use super::startup_cleanup_evidence::PersistentPromptCacheStartupCleanupEvidence;
use astronomical_runtime_integration::{MlxArray, MlxRuntime, PositionalFileReadMetrics};
use std::collections::HashMap;

const BLOCKS_DIRECTORY_NAME: &str = "blocks";
const VISUAL_EMBEDDINGS_DIRECTORY_NAME: &str = "visual_embeddings";
const SPECULATIVE_PREFILL_SELECTIONS_DIRECTORY_NAME: &str = "speculative_prefill_selections";
const SPECULATIVE_PREFILL_TARGET_STATES_DIRECTORY_NAME: &str = "speculative_prefill_target_states";

/// Valid enabled prompt-cache filesystem state for one active model under one global quota.
#[derive(Clone, Debug)]
pub struct PersistentPromptCacheDiskStoreConfig {
    active_model_prompt_cache_directory: PathBuf,
    pub(super) global_prompt_cache_root_directory: PathBuf,
    global_prompt_cache_maximum_size_bytes: u64,
}

impl PersistentPromptCacheDiskStoreConfig {
    #[must_use]
    pub fn new(
        active_model_prompt_cache_directory: PathBuf,
        global_prompt_cache_root_directory: PathBuf,
        global_prompt_cache_maximum_size_bytes: u64,
    ) -> Self {
        Self {
            active_model_prompt_cache_directory,
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
        }
    }

    /// Derives an isolated cache namespace while retaining the shared global quota.
    #[must_use]
    pub fn for_model(&self, model_id: &str, model_revision: &str) -> Self {
        Self::new(
            self.global_prompt_cache_root_directory
                .join(model_id)
                .join(model_revision),
            self.global_prompt_cache_root_directory.clone(),
            self.global_prompt_cache_maximum_size_bytes,
        )
    }

    #[must_use]
    pub const fn global_prompt_cache_maximum_size_bytes(&self) -> u64 {
        self.global_prompt_cache_maximum_size_bytes
    }
}

/// Persistent, descriptor-backed SSD cache for Qwen3.5-MoE prompt-cache files.
pub struct PersistentPromptCacheDiskStore {
    active_model_prompt_cache_directory: PathBuf,
    pub(super) blocks_directory: PathBuf,
    pub(crate) visual_embeddings_directory: PathBuf,
    pub(crate) speculative_prefill_selections_directory: PathBuf,
    pub(crate) speculative_prefill_target_states_directory: PathBuf,
    pub(crate) global_prompt_cache_root_directory: PathBuf,
    pub(crate) global_prompt_cache_maximum_size_bytes: u64,
    pub(crate) global_prompt_cache_total_size_bytes: AtomicU64,
    pub(crate) global_visual_embedding_total_size_bytes: AtomicU64,
    pub(crate) model_contract: PersistentPromptCacheModelContract,
    tracked_files: Mutex<PersistentPromptCacheDiskStoreIndex>,
    write_operations: Mutex<()>,
    pending_startup_cleanup_evidence: Mutex<Option<PersistentPromptCacheStartupCleanupEvidence>>,
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
        let blocks_directory = persistent_prompt_cache_directory.join(BLOCKS_DIRECTORY_NAME);
        let visual_embeddings_directory =
            persistent_prompt_cache_directory.join(VISUAL_EMBEDDINGS_DIRECTORY_NAME);
        let speculative_prefill_selections_directory =
            persistent_prompt_cache_directory.join(SPECULATIVE_PREFILL_SELECTIONS_DIRECTORY_NAME);
        let speculative_prefill_target_states_directory = persistent_prompt_cache_directory
            .join(SPECULATIVE_PREFILL_TARGET_STATES_DIRECTORY_NAME);
        let mut startup_cleanup_evidence = PersistentPromptCacheStartupCleanupEvidence::default();
        // Open order is a recovery protocol: establish trusted directories,
        // rebuild the active-model index from valid committed artifacts, remove
        // abandoned global transactions, then reconcile retention and quota.
        prepare_prompt_cache_directory_tree(
            &global_prompt_cache_root_directory,
            &persistent_prompt_cache_directory,
            &[
                &blocks_directory,
                &visual_embeddings_directory,
                &speculative_prefill_selections_directory,
                &speculative_prefill_target_states_directory,
            ],
        )?;
        let mut tracked_files = PersistentPromptCacheDiskStoreIndex::default();
        scan_current_format_block_directories(
            &blocks_directory,
            &mut tracked_files,
            &model_contract,
            &mut startup_cleanup_evidence,
        )?;
        scan_current_format_directory(
            &speculative_prefill_selections_directory,
            PersistentPromptCacheFileKind::SpeculativePrefillSelection,
            &mut tracked_files,
            &mut startup_cleanup_evidence,
            |file, file_path| {
                validate_current_file_header(
                    PersistentPromptCacheFileKind::SpeculativePrefillSelection,
                    file,
                    file_path,
                    &model_contract,
                )
                .is_ok()
            },
        )?;
        scan_current_format_directory(
            &speculative_prefill_target_states_directory,
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
            &mut tracked_files,
            &mut startup_cleanup_evidence,
            |file, file_path| {
                validate_current_file_header(
                    PersistentPromptCacheFileKind::SpeculativePrefillTargetState,
                    file,
                    file_path,
                    &model_contract,
                )
                .is_ok()
            },
        )?;
        let disk_store = Self {
            active_model_prompt_cache_directory: persistent_prompt_cache_directory,
            blocks_directory,
            visual_embeddings_directory,
            speculative_prefill_selections_directory,
            speculative_prefill_target_states_directory,
            global_prompt_cache_root_directory,
            global_prompt_cache_maximum_size_bytes,
            global_prompt_cache_total_size_bytes: AtomicU64::new(0),
            global_visual_embedding_total_size_bytes: AtomicU64::new(0),
            model_contract,
            tracked_files: Mutex::new(tracked_files),
            write_operations: Mutex::new(()),
            pending_startup_cleanup_evidence: Mutex::new(startup_cleanup_evidence.into_non_empty()),
        };
        // Stale bytes must disappear before quota considers deleting useful
        // content. Retention reconciliation then protects the validated active
        // chain while evicting unrelated content and completing crash cleanup.
        disk_store.remove_unconditionally_reclaimable_startup_artifacts()?;
        disk_store.reconcile_startup_retention_and_global_quota()?;
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

    pub fn speculative_prefill_selection_count(&self) -> usize {
        self.lock_tracked_files()
            .speculative_prefill_selection_count()
    }

    pub fn has_kv_block(&self, block_hash: &[u8; 32]) -> bool {
        self.tracked_file_still_exists(
            PersistentPromptCacheFileKind::SequenceStateBlock,
            block_hash,
        )
    }

    pub fn has_recurrent_snapshot(&self, block_hash: &[u8; 32]) -> bool {
        self.tracked_file_still_exists(
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            block_hash,
        )
    }

    pub fn has_visual_embedding(&self, visual_embedding_hash: &[u8; 32]) -> bool {
        self.tracked_file_still_exists(
            PersistentPromptCacheFileKind::VisualEmbedding,
            visual_embedding_hash,
        )
    }

    pub fn total_size_bytes(&self) -> u64 {
        self.global_prompt_cache_total_size_bytes
            .load(Ordering::Relaxed)
    }

    pub fn visual_embedding_total_size_bytes(&self) -> u64 {
        self.global_visual_embedding_total_size_bytes
            .load(Ordering::Relaxed)
    }

    pub fn startup_cleanup_evidence(&self) -> Option<PersistentPromptCacheStartupCleanupEvidence> {
        *self.lock_startup_cleanup_evidence()
    }

    pub fn take_startup_cleanup_evidence(
        &self,
    ) -> Option<PersistentPromptCacheStartupCleanupEvidence> {
        self.lock_startup_cleanup_evidence().take()
    }

    pub(crate) fn record_startup_cleanup_evidence(
        &self,
        additional_evidence: PersistentPromptCacheStartupCleanupEvidence,
    ) {
        if additional_evidence.into_non_empty().is_none() {
            return;
        }
        let mut pending_evidence = self.lock_startup_cleanup_evidence();
        match pending_evidence.as_mut() {
            Some(existing_evidence) => existing_evidence.merge(additional_evidence),
            None => *pending_evidence = Some(additional_evidence),
        }
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

    fn lock_startup_cleanup_evidence(
        &self,
    ) -> MutexGuard<'_, Option<PersistentPromptCacheStartupCleanupEvidence>> {
        self.pending_startup_cleanup_evidence
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn tracked_file_still_exists(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: &[u8; 32],
    ) -> bool {
        let tracked_file_path = self
            .lock_tracked_files()
            .file(
                persistent_prompt_cache_file_kind,
                persistent_prompt_cache_file_hash,
            )
            .map(|tracked_file| tracked_file.file_path.clone());
        let Some(tracked_file_path) = tracked_file_path else {
            return false;
        };
        // `NotFound` is authoritative and updates telemetry immediately. Other
        // metadata errors are left for the actual load path to report with full
        // typed context rather than turning a permission fault into a cache miss.
        match std::fs::symlink_metadata(&tracked_file_path) {
            Ok(_) => true,
            Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
                self.untrack_file_and_subtract_global_accounting(
                    persistent_prompt_cache_file_kind,
                    *persistent_prompt_cache_file_hash,
                );
                false
            }
            Err(_) => true,
        }
    }

    pub(crate) fn prepare_active_model_storage_directories(
        &self,
    ) -> Result<(), PersistentPromptCacheDiskStoreError> {
        prepare_prompt_cache_directory_tree(
            &self.global_prompt_cache_root_directory,
            &self.active_model_prompt_cache_directory,
            &[
                &self.blocks_directory,
                &self.visual_embeddings_directory,
                &self.speculative_prefill_selections_directory,
                &self.speculative_prefill_target_states_directory,
            ],
        )
    }

    pub fn load_kv_block(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        self.load_file_kind(
            runtime,
            persistent_prompt_cache_block_key,
            PersistentPromptCacheFileKind::SequenceStateBlock,
            positional_file_read_metrics,
        )
    }

    pub fn load_recurrent_snapshot(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        self.load_file_kind(
            runtime,
            persistent_prompt_cache_block_key,
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            positional_file_read_metrics,
        )
    }

    fn load_file_kind(
        &self,
        runtime: &MlxRuntime,
        persistent_prompt_cache_block_key: &PersistentPromptCacheBlockKey,
        file_kind: PersistentPromptCacheFileKind,
        positional_file_read_metrics: Option<Arc<PositionalFileReadMetrics>>,
    ) -> Result<Option<HashMap<String, MlxArray>>, PersistentPromptCacheDiskStoreError> {
        // Clone the path while holding the short index lock, then perform disk
        // and MLX work unlocked so unrelated cache lookups are not serialized.
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
        // Header validation happens before MLX maps payloads. This rejects wrong
        // model geometry and malformed offsets without allocating decoder arrays.
        let loaded_safetensors = runtime
            .load_safetensors(block_file, positional_file_read_metrics)
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
