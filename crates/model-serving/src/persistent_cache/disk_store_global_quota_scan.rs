//! Recursive global prompt-cache quota discovery.
//!
//! This module has no deletion authority. It takes one filesystem snapshot,
//! counts every owned byte, reconstructs committed block ancestry, and returns
//! deterministic eviction units. Keeping discovery separate from deletion
//! avoids rescanning and resorting the whole cache after every removed block.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::block_manifest::PersistentPromptCacheBlockManifest;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::parse_persistent_prompt_cache_file_hash_from_path;
use super::disk_store_global_quota_candidate::{
    GlobalPromptCacheBlockSubtree, GlobalPromptCacheEvictionCandidate, GlobalPromptCacheFile,
    GlobalPromptCacheStaleDirectory,
};

pub(super) struct GlobalPromptCacheQuotaScan {
    pub(super) eviction_candidates_oldest_written_first: Vec<GlobalPromptCacheEvictionCandidate>,
    pub(super) total_size_bytes: u64,
    pub(super) visual_embedding_total_size_bytes: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GlobalPromptCacheBlockIdentity {
    // A hash alone is not globally unique: separate model/revision directories
    // can contain the same token hash, and the same directory can retain stale
    // files from another tensor layout. All three fields are required before a
    // manifest parent edge may connect two blocks.
    blocks_directory_path: PathBuf,
    storage_contract_fingerprint: String,
    block_hash: [u8; 32],
}

#[derive(Debug)]
struct GlobalPromptCacheBlockDirectory {
    identity: GlobalPromptCacheBlockIdentity,
    parent_block_hash: Option<[u8; 32]>,
    block_directory_path: PathBuf,
    file_size_bytes: u64,
    modified_at: SystemTime,
    tracked_file_paths: Vec<PathBuf>,
}

pub(super) fn scan_global_prompt_cache_quota(
    global_prompt_cache_root_directory: &Path,
    excluded_directory: Option<&Path>,
) -> Result<GlobalPromptCacheQuotaScan, PersistentPromptCacheDiskStoreError> {
    let mut standalone_files = Vec::new();
    let mut stale_directories = Vec::new();
    let mut block_directories = Vec::new();
    scan_global_prompt_cache_entries(
        global_prompt_cache_root_directory,
        excluded_directory,
        &mut standalone_files,
        &mut stale_directories,
        &mut block_directories,
    )?;
    let mut total_size_bytes = 0_u64;
    let mut visual_embedding_total_size_bytes = 0_u64;
    for standalone_file in &standalone_files {
        total_size_bytes = checked_add_global_size(
            global_prompt_cache_root_directory,
            total_size_bytes,
            standalone_file.file_size_bytes,
        )?;
        if standalone_file.is_visual_embedding {
            visual_embedding_total_size_bytes = checked_add_global_size(
                global_prompt_cache_root_directory,
                visual_embedding_total_size_bytes,
                standalone_file.file_size_bytes,
            )?;
        }
    }
    for stale_directory in &stale_directories {
        total_size_bytes = checked_add_global_size(
            global_prompt_cache_root_directory,
            total_size_bytes,
            stale_directory.file_size_bytes,
        )?;
    }
    for block_directory in &block_directories {
        total_size_bytes = checked_add_global_size(
            global_prompt_cache_root_directory,
            total_size_bytes,
            block_directory.file_size_bytes,
        )?;
    }

    let mut eviction_candidates = standalone_files
        .into_iter()
        .map(GlobalPromptCacheEvictionCandidate::StandaloneFile)
        .chain(
            stale_directories
                .into_iter()
                .map(GlobalPromptCacheEvictionCandidate::StaleDirectory),
        )
        .collect::<Vec<_>>();
    eviction_candidates.extend(build_block_subtree_candidates(block_directories)?);
    // Stale transaction artifacts sort before durable content regardless of
    // age. Within each class, oldest-write-first gives predictable LRU-like
    // pressure relief without maintaining another persistent access database.
    eviction_candidates.sort_by(|left_candidate, right_candidate| {
        (!left_candidate.is_stale_transaction_artifact())
            .cmp(&(!right_candidate.is_stale_transaction_artifact()))
            .then_with(|| {
                left_candidate
                    .modified_at()
                    .cmp(&right_candidate.modified_at())
            })
            .then_with(|| {
                left_candidate
                    .tie_breaker_path()
                    .cmp(right_candidate.tie_breaker_path())
            })
    });

    Ok(GlobalPromptCacheQuotaScan {
        eviction_candidates_oldest_written_first: eviction_candidates,
        total_size_bytes,
        visual_embedding_total_size_bytes,
    })
}

fn scan_global_prompt_cache_entries(
    global_prompt_cache_root_directory: &Path,
    excluded_directory: Option<&Path>,
    standalone_files: &mut Vec<GlobalPromptCacheFile>,
    stale_directories: &mut Vec<GlobalPromptCacheStaleDirectory>,
    block_directories: &mut Vec<GlobalPromptCacheBlockDirectory>,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    let mut pending_directories = vec![global_prompt_cache_root_directory.to_path_buf()];
    while let Some(pending_directory) = pending_directories.pop() {
        if excluded_directory.is_some_and(|directory| pending_directory == directory) {
            continue;
        }
        let directory_entries = fs::read_dir(&pending_directory).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: pending_directory.clone(),
                source,
            }
        })?;
        for directory_entry_result in directory_entries {
            let directory_entry = directory_entry_result.map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                    persistent_prompt_cache_directory: pending_directory.clone(),
                    source,
                }
            })?;
            let entry_path = directory_entry.path();
            if excluded_directory.is_some_and(|directory| entry_path == directory) {
                continue;
            }
            let entry_file_type = directory_entry.file_type().map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: entry_path.clone(),
                    source,
                }
            })?;
            if entry_file_type.is_dir() {
                // A valid block directory is a leaf for this traversal: its
                // contents are counted together by `scan_block_directory`.
                if is_stale_block_staging_directory(&entry_path) {
                    stale_directories.push(scan_stale_directory(&entry_path)?);
                } else if let Some(block_directory) = scan_block_directory(&entry_path)? {
                    block_directories.push(block_directory);
                } else {
                    pending_directories.push(entry_path);
                }
            } else {
                standalone_files.push(scan_standalone_file(&entry_path)?);
            }
        }
    }
    Ok(())
}

fn scan_standalone_file(
    file_path: &Path,
) -> Result<GlobalPromptCacheFile, PersistentPromptCacheDiskStoreError> {
    let file_metadata = fs::symlink_metadata(file_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: file_path.to_path_buf(),
            source,
        }
    })?;
    let modified_at = file_metadata.modified().map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: file_path.to_path_buf(),
            source,
        }
    })?;
    let is_visual_embedding = file_path.parent().is_some_and(|parent_directory| {
        parent_directory
            .file_name()
            .is_some_and(|directory_name| directory_name == "visual_embeddings")
    });
    // `kv_blocks` and `recurrent_snapshots` are retired pre-format-11 storage
    // trees. Their files are recoverable cache artifacts, not valid committed
    // blocks, so startup may reclaim them before current-format content.
    let is_stale_writer_temp_file = file_path
        .extension()
        .is_some_and(|extension| extension == "tmp")
        || file_path.ancestors().any(|ancestor_path| {
            ancestor_path.file_name().is_some_and(|directory_name| {
                directory_name == "kv_blocks" || directory_name == "recurrent_snapshots"
            })
        });
    Ok(GlobalPromptCacheFile {
        file_path: file_path.to_path_buf(),
        file_size_bytes: file_metadata.len(),
        modified_at,
        is_visual_embedding,
        is_stale_writer_temp_file,
    })
}

fn scan_stale_directory(
    directory_path: &Path,
) -> Result<GlobalPromptCacheStaleDirectory, PersistentPromptCacheDiskStoreError> {
    let directory_metadata = fs::symlink_metadata(directory_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    let modified_at = directory_metadata.modified().map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    let (file_size_bytes, tracked_file_paths) = directory_file_size_and_paths(directory_path)?;
    Ok(GlobalPromptCacheStaleDirectory {
        directory_path: directory_path.to_path_buf(),
        file_size_bytes,
        modified_at,
        tracked_file_paths,
    })
}

fn scan_block_directory(
    directory_path: &Path,
) -> Result<Option<GlobalPromptCacheBlockDirectory>, PersistentPromptCacheDiskStoreError> {
    let Some(blocks_directory_path) = directory_path.parent().filter(|parent_directory| {
        parent_directory
            .file_name()
            .is_some_and(|directory_name| directory_name == "blocks")
    }) else {
        return Ok(None);
    };
    let Some(block_hash) = parse_persistent_prompt_cache_file_hash_from_path(directory_path) else {
        return Ok(None);
    };
    let directory_metadata = fs::symlink_metadata(directory_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    let modified_at = directory_metadata.modified().map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    // Global quota scans every model and revision, so it cannot validate a
    // foreign manifest against the active model contract. It reads only the
    // topology fields needed for safe grouping. Invalid manifests receive a
    // path-unique synthetic fingerprint, preventing accidental ancestry joins.
    let block_manifest =
        PersistentPromptCacheBlockManifest::read_unvalidated_from_block_directory(directory_path)
            .ok()
            .filter(|block_manifest| block_manifest.block_hash().ok() == Some(block_hash));
    let parent_block_hash = block_manifest
        .as_ref()
        .and_then(PersistentPromptCacheBlockManifest::parent_block_hash);
    let storage_contract_fingerprint = block_manifest.map_or_else(
        || format!("invalid-manifest:{}", directory_path.display()),
        |block_manifest| block_manifest.storage_contract_fingerprint().to_owned(),
    );
    let (file_size_bytes, tracked_file_paths) = directory_file_size_and_paths(directory_path)?;
    Ok(Some(GlobalPromptCacheBlockDirectory {
        identity: GlobalPromptCacheBlockIdentity {
            blocks_directory_path: blocks_directory_path.to_path_buf(),
            storage_contract_fingerprint,
            block_hash,
        },
        parent_block_hash,
        block_directory_path: directory_path.to_path_buf(),
        file_size_bytes,
        modified_at,
        tracked_file_paths,
    }))
}

fn build_block_subtree_candidates(
    block_directories: Vec<GlobalPromptCacheBlockDirectory>,
) -> Result<Vec<GlobalPromptCacheEvictionCandidate>, PersistentPromptCacheDiskStoreError> {
    let block_directory_by_identity = block_directories
        .into_iter()
        .map(|block_directory| (block_directory.identity.clone(), block_directory))
        .collect::<HashMap<_, _>>();
    // Build parent -> children edges only inside one blocks directory and one
    // storage fingerprint. This is the core guard against cross-model eviction.
    let mut children_by_parent_identity = HashMap::<GlobalPromptCacheBlockIdentity, Vec<_>>::new();
    for block_directory in block_directory_by_identity.values() {
        let Some(parent_block_hash) = block_directory.parent_block_hash else {
            continue;
        };
        let parent_identity = GlobalPromptCacheBlockIdentity {
            blocks_directory_path: block_directory.identity.blocks_directory_path.clone(),
            storage_contract_fingerprint: block_directory
                .identity
                .storage_contract_fingerprint
                .clone(),
            block_hash: parent_block_hash,
        };
        children_by_parent_identity
            .entry(parent_identity)
            .or_default()
            .push(block_directory.identity.clone());
    }
    let mut eviction_candidates = Vec::new();
    // Every block can be a candidate root. Overlapping candidates are expected;
    // the deletion owner records removed paths and skips later overlaps.
    for block_identity in block_directory_by_identity.keys() {
        let mut subtree_block_identities = Vec::new();
        collect_subtree_block_identities(
            block_identity,
            &children_by_parent_identity,
            &mut subtree_block_identities,
        );
        let mut subtree_file_size_bytes = 0_u64;
        let mut subtree_block_directory_paths = Vec::new();
        let mut subtree_tracked_file_paths = Vec::new();
        for subtree_block_identity in subtree_block_identities {
            let Some(subtree_block_directory) =
                block_directory_by_identity.get(&subtree_block_identity)
            else {
                continue;
            };
            subtree_file_size_bytes = subtree_file_size_bytes
                .checked_add(subtree_block_directory.file_size_bytes)
                .ok_or_else(|| {
                    PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                        global_prompt_cache_root_directory: subtree_block_directory
                            .identity
                            .blocks_directory_path
                            .clone(),
                    }
                })?;
            subtree_block_directory_paths
                .push(subtree_block_directory.block_directory_path.clone());
            subtree_tracked_file_paths.extend(subtree_block_directory.tracked_file_paths.clone());
        }
        let root_block_directory = block_directory_by_identity
            .get(block_identity)
            .expect("the root block identity came from the block map");
        eviction_candidates.push(GlobalPromptCacheEvictionCandidate::BlockSubtree(
            GlobalPromptCacheBlockSubtree {
                root_block_directory_path: root_block_directory.block_directory_path.clone(),
                file_size_bytes: subtree_file_size_bytes,
                modified_at: root_block_directory.modified_at,
                block_directory_paths: subtree_block_directory_paths,
                tracked_file_paths: subtree_tracked_file_paths,
            },
        ));
    }
    Ok(eviction_candidates)
}

fn collect_subtree_block_identities(
    root_block_identity: &GlobalPromptCacheBlockIdentity,
    children_by_parent_identity: &HashMap<
        GlobalPromptCacheBlockIdentity,
        Vec<GlobalPromptCacheBlockIdentity>,
    >,
    subtree_block_identities: &mut Vec<GlobalPromptCacheBlockIdentity>,
) {
    // Use an explicit stack and visited set. Corrupt manifests can form cycles,
    // and quota recovery must terminate rather than recurse forever or overflow.
    let mut pending_block_identities = vec![root_block_identity.clone()];
    let mut visited_block_identities = HashSet::new();
    while let Some(block_identity) = pending_block_identities.pop() {
        if !visited_block_identities.insert(block_identity.clone()) {
            continue;
        }
        subtree_block_identities.push(block_identity.clone());
        if let Some(child_block_identities) = children_by_parent_identity.get(&block_identity) {
            pending_block_identities.extend(child_block_identities.iter().cloned());
        }
    }
}

fn directory_file_size_and_paths(
    directory_path: &Path,
) -> Result<(u64, Vec<PathBuf>), PersistentPromptCacheDiskStoreError> {
    let mut pending_directories = vec![directory_path.to_path_buf()];
    let mut total_size_bytes = 0_u64;
    let mut tracked_file_paths = Vec::new();
    while let Some(pending_directory) = pending_directories.pop() {
        let directory_entries = fs::read_dir(&pending_directory).map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: pending_directory.clone(),
                source,
            }
        })?;
        for directory_entry_result in directory_entries {
            let directory_entry = directory_entry_result.map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                    persistent_prompt_cache_directory: pending_directory.clone(),
                    source,
                }
            })?;
            let entry_path = directory_entry.path();
            let entry_file_type = directory_entry.file_type().map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: entry_path.clone(),
                    source,
                }
            })?;
            if entry_file_type.is_dir() {
                pending_directories.push(entry_path);
                continue;
            }
            let file_metadata = fs::symlink_metadata(&entry_path).map_err(|source| {
                PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: entry_path.clone(),
                    source,
                }
            })?;
            total_size_bytes = total_size_bytes
                .checked_add(file_metadata.len())
                .ok_or_else(|| {
                    PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                        global_prompt_cache_root_directory: directory_path.to_path_buf(),
                    }
                })?;
            tracked_file_paths.push(entry_path);
        }
    }
    Ok((total_size_bytes, tracked_file_paths))
}

fn checked_add_global_size(
    global_prompt_cache_root_directory: &Path,
    accumulated_size_bytes: u64,
    additional_size_bytes: u64,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    accumulated_size_bytes
        .checked_add(additional_size_bytes)
        .ok_or_else(
            || PersistentPromptCacheDiskStoreError::GlobalPromptCacheSizeOverflow {
                global_prompt_cache_root_directory: global_prompt_cache_root_directory
                    .to_path_buf(),
            },
        )
}

fn is_stale_block_staging_directory(directory_path: &Path) -> bool {
    directory_path.parent().is_some_and(|parent_directory| {
        parent_directory
            .file_name()
            .is_some_and(|directory_name| directory_name == "blocks")
    }) && directory_path
        .file_name()
        .and_then(|directory_name| directory_name.to_str())
        .is_some_and(|directory_name| directory_name.contains(".staging-"))
}
