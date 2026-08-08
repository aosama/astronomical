use std::collections::HashMap;
use std::path::PathBuf;

use super::disk_store_file::PersistentPromptCacheFileKind;

#[derive(Clone, Debug)]
pub(crate) struct TrackedPersistentPromptCacheFile {
    pub(crate) file_path: PathBuf,
    pub(crate) file_size_bytes: u64,
}

#[derive(Default)]
pub(crate) struct PersistentPromptCacheDiskStoreIndex {
    kv_blocks: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    recurrent_snapshots: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    visual_embeddings: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    speculative_prefill_selections: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    speculative_prefill_target_states: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
}

impl PersistentPromptCacheDiskStoreIndex {
    pub(super) fn sequence_state_block_count(&self) -> usize {
        self.kv_blocks.len()
    }

    pub(super) fn boundary_state_snapshot_count(&self) -> usize {
        self.recurrent_snapshots.len()
    }

    pub(super) fn visual_embedding_count(&self) -> usize {
        self.visual_embeddings.len()
    }

    pub(super) fn speculative_prefill_selection_count(&self) -> usize {
        self.speculative_prefill_selections.len()
    }

    pub(crate) fn file(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: &[u8; 32],
    ) -> Option<&TrackedPersistentPromptCacheFile> {
        match persistent_prompt_cache_file_kind {
            PersistentPromptCacheFileKind::SequenceStateBlock => {
                self.kv_blocks.get(persistent_prompt_cache_file_hash)
            }
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => self
                .recurrent_snapshots
                .get(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::VisualEmbedding => self
                .visual_embeddings
                .get(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::SpeculativePrefillSelection => self
                .speculative_prefill_selections
                .get(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState => self
                .speculative_prefill_target_states
                .get(persistent_prompt_cache_file_hash),
        }
    }

    pub(crate) fn insert_file(
        &mut self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: [u8; 32],
        tracked_persistent_prompt_cache_file: TrackedPersistentPromptCacheFile,
    ) {
        match persistent_prompt_cache_file_kind {
            PersistentPromptCacheFileKind::SequenceStateBlock => {
                self.kv_blocks.insert(
                    persistent_prompt_cache_file_hash,
                    tracked_persistent_prompt_cache_file,
                );
            }
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
                self.recurrent_snapshots.insert(
                    persistent_prompt_cache_file_hash,
                    tracked_persistent_prompt_cache_file,
                );
            }
            PersistentPromptCacheFileKind::VisualEmbedding => {
                self.visual_embeddings.insert(
                    persistent_prompt_cache_file_hash,
                    tracked_persistent_prompt_cache_file,
                );
            }
            PersistentPromptCacheFileKind::SpeculativePrefillSelection => {
                self.speculative_prefill_selections.insert(
                    persistent_prompt_cache_file_hash,
                    tracked_persistent_prompt_cache_file,
                );
            }
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState => {
                self.speculative_prefill_target_states.insert(
                    persistent_prompt_cache_file_hash,
                    tracked_persistent_prompt_cache_file,
                );
            }
        }
    }

    pub(crate) fn remove_file(
        &mut self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: &[u8; 32],
    ) -> Option<TrackedPersistentPromptCacheFile> {
        match persistent_prompt_cache_file_kind {
            PersistentPromptCacheFileKind::SequenceStateBlock => {
                self.kv_blocks.remove(persistent_prompt_cache_file_hash)
            }
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => self
                .recurrent_snapshots
                .remove(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::VisualEmbedding => self
                .visual_embeddings
                .remove(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::SpeculativePrefillSelection => self
                .speculative_prefill_selections
                .remove(persistent_prompt_cache_file_hash),
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState => self
                .speculative_prefill_target_states
                .remove(persistent_prompt_cache_file_hash),
        }
    }

    pub(super) fn remove_files_by_path(&mut self, removed_file_paths: &[PathBuf]) {
        self.kv_blocks
            .retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
        self.recurrent_snapshots
            .retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
        self.visual_embeddings
            .retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
        self.speculative_prefill_selections
            .retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
        self.speculative_prefill_target_states
            .retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
    }

    pub(super) fn recurrent_snapshot_file_size_bytes(
        &self,
        recurrent_snapshot_hash: &[u8; 32],
    ) -> Option<u64> {
        self.recurrent_snapshots
            .get(recurrent_snapshot_hash)
            .map(|tracked_file| tracked_file.file_size_bytes)
    }
}
