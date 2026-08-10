//! Typed eviction units produced by the global cache scan.
//!
//! Committed prompt-cache blocks are never represented as independent files.
//! They are grouped into ancestry-closed subtrees so eviction cannot leave a
//! descendant whose required sequence-state parent has disappeared.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug)]
pub(super) struct GlobalPromptCacheFile {
    pub(super) file_path: PathBuf,
    pub(super) file_size_bytes: u64,
    pub(super) modified_at: SystemTime,
    pub(super) is_visual_embedding: bool,
    pub(super) cleanup_classification: GlobalPromptCacheStandaloneFileClassification,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlobalPromptCacheStandaloneFileClassification {
    AbandonedTransaction,
    ObsoleteFormat,
    CommittedArtifact,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlobalPromptCacheCleanupClassification {
    InterruptedTransactionRecovery,
    ObsoleteFormat,
    QuotaEviction,
}

#[derive(Debug)]
pub(super) struct GlobalPromptCacheStaleDirectory {
    pub(super) directory_path: PathBuf,
    pub(super) file_size_bytes: u64,
    pub(super) modified_at: SystemTime,
    pub(super) tracked_file_paths: Vec<PathBuf>,
}

#[derive(Debug)]
pub(super) struct GlobalPromptCacheBlockSubtree {
    pub(super) root_block_directory_path: PathBuf,
    pub(super) file_size_bytes: u64,
    pub(super) modified_at: SystemTime,
    pub(super) block_directory_paths: Vec<PathBuf>,
    pub(super) tracked_file_paths: Vec<PathBuf>,
}

#[derive(Debug)]
pub(super) enum GlobalPromptCacheEvictionCandidate {
    /// A non-block artifact such as a visual embedding or legacy temporary file.
    StandaloneFile(GlobalPromptCacheFile),
    /// An abandoned `.staging-*` transaction, always removable before content.
    StaleDirectory(GlobalPromptCacheStaleDirectory),
    /// One committed block and every descendant in the same storage namespace.
    BlockSubtree(GlobalPromptCacheBlockSubtree),
}

impl GlobalPromptCacheEvictionCandidate {
    pub(super) fn modified_at(&self) -> SystemTime {
        match self {
            Self::StandaloneFile(global_prompt_cache_file) => global_prompt_cache_file.modified_at,
            Self::StaleDirectory(global_prompt_cache_stale_directory) => {
                global_prompt_cache_stale_directory.modified_at
            }
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                global_prompt_cache_block_subtree.modified_at
            }
        }
    }

    pub(super) fn tie_breaker_path(&self) -> &Path {
        // Filesystem timestamps may have coarse resolution. A stable path tie
        // breaker makes eviction deterministic across scans of identical state.
        match self {
            Self::StandaloneFile(global_prompt_cache_file) => &global_prompt_cache_file.file_path,
            Self::StaleDirectory(global_prompt_cache_stale_directory) => {
                &global_prompt_cache_stale_directory.directory_path
            }
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                &global_prompt_cache_block_subtree.root_block_directory_path
            }
        }
    }

    pub(super) fn unconditional_cleanup_classification(
        &self,
    ) -> Option<GlobalPromptCacheCleanupClassification> {
        match self {
            Self::StandaloneFile(global_prompt_cache_file) => {
                match global_prompt_cache_file.cleanup_classification {
                    GlobalPromptCacheStandaloneFileClassification::AbandonedTransaction => {
                        Some(GlobalPromptCacheCleanupClassification::InterruptedTransactionRecovery)
                    }
                    GlobalPromptCacheStandaloneFileClassification::ObsoleteFormat => {
                        Some(GlobalPromptCacheCleanupClassification::ObsoleteFormat)
                    }
                    GlobalPromptCacheStandaloneFileClassification::CommittedArtifact => None,
                }
            }
            Self::StaleDirectory(_) => {
                Some(GlobalPromptCacheCleanupClassification::InterruptedTransactionRecovery)
            }
            Self::BlockSubtree(_) => None,
        }
    }

    pub(super) fn is_unconditionally_removable(&self) -> bool {
        self.unconditional_cleanup_classification().is_some()
    }

    pub(super) fn removed_artifact_count(&self) -> usize {
        usize::from(matches!(self, Self::StandaloneFile(_)))
    }

    pub(super) fn removed_block_count(&self) -> usize {
        match self {
            Self::StandaloneFile(_) => 0,
            Self::StaleDirectory(_) => 1,
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                global_prompt_cache_block_subtree
                    .block_directory_paths
                    .len()
            }
        }
    }

    pub(super) fn file_size_bytes(&self) -> u64 {
        match self {
            Self::StandaloneFile(global_prompt_cache_file) => {
                global_prompt_cache_file.file_size_bytes
            }
            Self::StaleDirectory(global_prompt_cache_stale_directory) => {
                global_prompt_cache_stale_directory.file_size_bytes
            }
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                global_prompt_cache_block_subtree.file_size_bytes
            }
        }
    }

    pub(super) fn visual_embedding_size_bytes(&self) -> u64 {
        if let Self::StandaloneFile(global_prompt_cache_file) = self
            && global_prompt_cache_file.is_visual_embedding
        {
            global_prompt_cache_file.file_size_bytes
        } else {
            0
        }
    }

    pub(super) fn tracked_file_paths(&self) -> &[PathBuf] {
        match self {
            Self::StandaloneFile(global_prompt_cache_file) => {
                std::slice::from_ref(&global_prompt_cache_file.file_path)
            }
            Self::StaleDirectory(global_prompt_cache_stale_directory) => {
                &global_prompt_cache_stale_directory.tracked_file_paths
            }
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                &global_prompt_cache_block_subtree.tracked_file_paths
            }
        }
    }

    pub(super) fn block_directory_paths(&self) -> &[PathBuf] {
        match self {
            Self::BlockSubtree(global_prompt_cache_block_subtree) => {
                &global_prompt_cache_block_subtree.block_directory_paths
            }
            _ => &[],
        }
    }

    pub(super) fn contains_protected_block_directory(
        &self,
        protected_block_directory_paths: &[PathBuf],
    ) -> bool {
        // Protect the whole candidate if any member intersects the active chain;
        // deleting only its unprotected members would violate subtree atomicity.
        let Self::BlockSubtree(global_prompt_cache_block_subtree) = self else {
            return false;
        };
        global_prompt_cache_block_subtree
            .block_directory_paths
            .iter()
            .any(|block_directory_path| {
                protected_block_directory_paths.contains(block_directory_path)
            })
    }
}
