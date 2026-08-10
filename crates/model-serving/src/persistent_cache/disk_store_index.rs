//! Process-local index of prompt-cache files already validated against disk.
//!
//! This index accelerates lookup and exposes counters, but it is never durable
//! authority. Publication updates it only after commit; startup rebuilds it from
//! disk; and read paths remove entries whose files disappeared concurrently.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use super::disk_store_file::PersistentPromptCacheFileKind;

#[derive(Clone, Debug)]
pub(crate) struct TrackedPersistentPromptCacheFile {
    pub(crate) file_path: PathBuf,
    pub(crate) file_size_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct TrackedPersistentPromptCacheBlock {
    // One entry represents the directory as a topology unit. State files remain
    // optional because retention may remove a redundant parent boundary while
    // preserving its required sequence state and ancestry metadata.
    pub(crate) block_directory_path: PathBuf,
    pub(crate) block_index: u32,
    pub(crate) parent_block_hash: Option<[u8; 32]>,
    pub(crate) sequence_state_file: Option<TrackedPersistentPromptCacheFile>,
    pub(crate) boundary_state_file: Option<TrackedPersistentPromptCacheFile>,
}

#[derive(Default)]
pub(crate) struct PersistentPromptCacheDiskStoreIndex {
    blocks: HashMap<[u8; 32], TrackedPersistentPromptCacheBlock>,
    visual_embeddings: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    speculative_prefill_selections: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    speculative_prefill_target_states: HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
}

impl PersistentPromptCacheDiskStoreIndex {
    pub(super) fn sequence_state_block_count(&self) -> usize {
        self.blocks
            .values()
            .filter(|tracked_block| tracked_block.sequence_state_file.is_some())
            .count()
    }

    pub(super) fn boundary_state_snapshot_count(&self) -> usize {
        self.blocks
            .values()
            .filter(|tracked_block| tracked_block.boundary_state_file.is_some())
            .count()
    }

    pub(super) fn visual_embedding_count(&self) -> usize {
        self.visual_embeddings.len()
    }

    pub(super) fn speculative_prefill_selection_count(&self) -> usize {
        self.speculative_prefill_selections.len()
    }

    pub(crate) fn block(
        &self,
        block_hash: &[u8; 32],
    ) -> Option<&TrackedPersistentPromptCacheBlock> {
        self.blocks.get(block_hash)
    }

    pub(crate) fn insert_block(
        &mut self,
        block_hash: [u8; 32],
        tracked_block: TrackedPersistentPromptCacheBlock,
    ) {
        self.blocks.insert(block_hash, tracked_block);
    }

    pub(crate) fn tracked_blocks(&self) -> Vec<([u8; 32], TrackedPersistentPromptCacheBlock)> {
        self.blocks
            .iter()
            .map(|(block_hash, tracked_block)| (*block_hash, tracked_block.clone()))
            .collect()
    }

    pub(crate) fn remove_block(
        &mut self,
        block_hash: &[u8; 32],
    ) -> Option<TrackedPersistentPromptCacheBlock> {
        self.blocks.remove(block_hash)
    }

    pub(crate) fn protected_ancestry_directory_paths(
        &self,
        chain_tip_block_hash: [u8; 32],
    ) -> Vec<PathBuf> {
        // Walk from tip to root, stopping on missing or cyclic topology. Startup
        // validation should have removed both, but quota protection must remain
        // bounded even if disk changes after the index was built.
        let mut protected_block_directory_paths = Vec::new();
        let mut visited_block_hashes = HashSet::new();
        let mut next_block_hash = Some(chain_tip_block_hash);
        while let Some(block_hash) = next_block_hash {
            if !visited_block_hashes.insert(block_hash) {
                break;
            }
            let Some(tracked_block) = self.blocks.get(&block_hash) else {
                break;
            };
            protected_block_directory_paths.push(tracked_block.block_directory_path.clone());
            next_block_hash = tracked_block.parent_block_hash;
        }
        protected_block_directory_paths
    }

    pub(crate) fn file(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
        persistent_prompt_cache_file_hash: &[u8; 32],
    ) -> Option<&TrackedPersistentPromptCacheFile> {
        match persistent_prompt_cache_file_kind {
            PersistentPromptCacheFileKind::SequenceStateBlock => self
                .blocks
                .get(persistent_prompt_cache_file_hash)
                .and_then(|tracked_block| tracked_block.sequence_state_file.as_ref()),
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => self
                .blocks
                .get(persistent_prompt_cache_file_hash)
                .and_then(|tracked_block| tracked_block.boundary_state_file.as_ref()),
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
                if let Some(tracked_block) = self.blocks.get_mut(&persistent_prompt_cache_file_hash)
                {
                    tracked_block.sequence_state_file = Some(tracked_persistent_prompt_cache_file);
                }
            }
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
                if let Some(tracked_block) = self.blocks.get_mut(&persistent_prompt_cache_file_hash)
                {
                    tracked_block.boundary_state_file = Some(tracked_persistent_prompt_cache_file);
                }
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
            PersistentPromptCacheFileKind::SequenceStateBlock => self
                .blocks
                .get_mut(persistent_prompt_cache_file_hash)
                .and_then(|tracked_block| tracked_block.sequence_state_file.take()),
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => self
                .blocks
                .get_mut(persistent_prompt_cache_file_hash)
                .and_then(|tracked_block| tracked_block.boundary_state_file.take()),
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

    pub(crate) fn files(
        &self,
        persistent_prompt_cache_file_kind: PersistentPromptCacheFileKind,
    ) -> Vec<([u8; 32], TrackedPersistentPromptCacheFile)> {
        match persistent_prompt_cache_file_kind {
            PersistentPromptCacheFileKind::SequenceStateBlock => self
                .blocks
                .iter()
                .filter_map(|(block_hash, tracked_block)| {
                    tracked_block
                        .sequence_state_file
                        .as_ref()
                        .map(|tracked_file| (*block_hash, tracked_file.clone()))
                })
                .collect(),
            PersistentPromptCacheFileKind::BoundaryStateSnapshot => self
                .blocks
                .iter()
                .filter_map(|(block_hash, tracked_block)| {
                    tracked_block
                        .boundary_state_file
                        .as_ref()
                        .map(|tracked_file| (*block_hash, tracked_file.clone()))
                })
                .collect(),
            PersistentPromptCacheFileKind::VisualEmbedding => {
                clone_file_map(&self.visual_embeddings)
            }
            PersistentPromptCacheFileKind::SpeculativePrefillSelection => {
                clone_file_map(&self.speculative_prefill_selections)
            }
            PersistentPromptCacheFileKind::SpeculativePrefillTargetState => {
                clone_file_map(&self.speculative_prefill_target_states)
            }
        }
    }

    pub(super) fn remove_files_by_path(&mut self, removed_file_paths: &[PathBuf]) {
        // Global quota scans all model namespaces and returns paths rather than
        // active-model hashes. Reconcile those paths back into this local index.
        for tracked_block in self.blocks.values_mut() {
            if tracked_block
                .sequence_state_file
                .as_ref()
                .is_some_and(|tracked_file| removed_file_paths.contains(&tracked_file.file_path))
            {
                tracked_block.sequence_state_file = None;
            }
            if tracked_block
                .boundary_state_file
                .as_ref()
                .is_some_and(|tracked_file| removed_file_paths.contains(&tracked_file.file_path))
            {
                tracked_block.boundary_state_file = None;
            }
        }
        retain_files_not_removed(&mut self.visual_embeddings, removed_file_paths);
        retain_files_not_removed(&mut self.speculative_prefill_selections, removed_file_paths);
        retain_files_not_removed(
            &mut self.speculative_prefill_target_states,
            removed_file_paths,
        );
    }

    pub(super) fn remove_blocks_by_directory_paths(&mut self, removed_directory_paths: &[PathBuf]) {
        // Remove complete topology entries only for subtree-directory eviction;
        // deleting one retained state file uses `remove_files_by_path` instead.
        self.blocks.retain(|_, tracked_block| {
            !removed_directory_paths.contains(&tracked_block.block_directory_path)
        });
    }
}

fn clone_file_map(
    tracked_files: &HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
) -> Vec<([u8; 32], TrackedPersistentPromptCacheFile)> {
    tracked_files
        .iter()
        .map(|(file_hash, tracked_file)| (*file_hash, tracked_file.clone()))
        .collect()
}

fn retain_files_not_removed(
    tracked_files: &mut HashMap<[u8; 32], TrackedPersistentPromptCacheFile>,
    removed_file_paths: &[PathBuf],
) {
    tracked_files.retain(|_, tracked_file| !removed_file_paths.contains(&tracked_file.file_path));
}
