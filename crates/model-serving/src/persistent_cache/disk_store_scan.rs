//! Startup scan: discovers, validates, and cleans persistent prompt-cache files
//! during `PersistentPromptCacheDiskStore::open()`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::block_manifest::{
    BOUNDARY_STATE_FILE_NAME, PersistentPromptCacheBlockManifest, SEQUENCE_STATE_FILE_NAME,
};
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::disk_store_file::{
    PersistentPromptCacheFileKind, open_without_following_symlinks,
    parse_persistent_prompt_cache_file_hash_from_path,
    remove_cache_owned_directory_or_confirm_absent, remove_cache_owned_file_or_confirm_absent,
    validate_current_file_header,
};
use super::disk_store_index::{
    PersistentPromptCacheDiskStoreIndex, TrackedPersistentPromptCacheBlock,
    TrackedPersistentPromptCacheFile,
};
use super::model_contract::PersistentPromptCacheModelContract;
use super::retention_policy::persistent_prompt_cache_boundary_is_common_prefix_checkpoint;
use super::startup_cleanup_evidence::PersistentPromptCacheStartupCleanupEvidence;

pub(crate) fn scan_current_format_directory<HeaderValidator>(
    directory: &Path,
    file_kind: PersistentPromptCacheFileKind,
    tracked_files: &mut PersistentPromptCacheDiskStoreIndex,
    startup_cleanup_evidence: &mut PersistentPromptCacheStartupCleanupEvidence,
    header_validator: HeaderValidator,
) -> Result<(), PersistentPromptCacheDiskStoreError>
where
    HeaderValidator: Fn(&std::fs::File, &Path) -> bool,
{
    let directory_entries = std::fs::read_dir(directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
            persistent_prompt_cache_directory: directory.to_path_buf(),
            source,
        }
    })?;
    for directory_entry_result in directory_entries {
        let directory_entry = directory_entry_result.map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: directory.to_path_buf(),
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
        if entry_path
            .extension()
            .is_some_and(|extension| extension == "tmp")
        {
            let removed_byte_count = cache_owned_file_byte_count(&entry_path)?;
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            startup_cleanup_evidence
                .interrupted_transaction_recovery
                .record_artifact(removed_byte_count);
            continue;
        }
        if !entry_file_type.is_file()
            || entry_path
                .extension()
                .is_none_or(|ext| ext != "safetensors")
        {
            continue;
        }
        let Some(persistent_prompt_cache_file_hash) =
            parse_persistent_prompt_cache_file_hash_from_path(&entry_path)
        else {
            let removed_byte_count = cache_owned_file_byte_count(&entry_path)?;
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_artifact(removed_byte_count);
            continue;
        };
        let file_size_bytes = std::fs::symlink_metadata(&entry_path)
            .map(|metadata| metadata.len())
            .map_err(
                |source| PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                    block_file_path: entry_path.clone(),
                    source,
                },
            )?;
        let file = open_without_following_symlinks(&entry_path).map_err(|source| {
            PersistentPromptCacheDiskStoreError::OpenBlockFile {
                block_file_path: entry_path.clone(),
                source,
            }
        })?;
        if !header_validator(&file, &entry_path) {
            remove_cache_owned_file_or_confirm_absent(&entry_path)?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_artifact(file_size_bytes);
            continue;
        }
        tracked_files.insert_file(
            file_kind,
            persistent_prompt_cache_file_hash,
            TrackedPersistentPromptCacheFile {
                file_path: entry_path,
                file_size_bytes,
            },
        );
    }
    Ok(())
}

pub(crate) fn scan_current_format_block_directories(
    blocks_directory: &Path,
    tracked_files: &mut PersistentPromptCacheDiskStoreIndex,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    startup_cleanup_evidence: &mut PersistentPromptCacheStartupCleanupEvidence,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    // Phase one validates each directory in isolation and builds candidates.
    // Phase two below validates cross-directory ancestry before indexing any of
    // them, preventing lookup from observing a partially accepted graph.
    let mut block_candidate_by_hash = HashMap::new();
    let directory_entries = std::fs::read_dir(blocks_directory).map_err(|source| {
        PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
            persistent_prompt_cache_directory: blocks_directory.to_path_buf(),
            source,
        }
    })?;
    for directory_entry_result in directory_entries {
        let directory_entry = directory_entry_result.map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadPromptCacheDirectory {
                persistent_prompt_cache_directory: blocks_directory.to_path_buf(),
                source,
            }
        })?;
        let block_directory_path = directory_entry.path();
        let entry_file_type = directory_entry.file_type().map_err(|source| {
            PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: block_directory_path.clone(),
                source,
            }
        })?;
        if !entry_file_type.is_dir() {
            continue;
        }
        let Some(block_directory_name) = block_directory_path
            .file_name()
            .and_then(|name| name.to_str())
        else {
            let removed_byte_count = cache_owned_directory_byte_count(&block_directory_path)?;
            remove_cache_owned_directory_or_confirm_absent(&block_directory_path)?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_block(removed_byte_count);
            continue;
        };
        // A staging name proves publication never reached its atomic directory
        // rename. Remove the whole transaction; individual files are not salvageable.
        if block_directory_name.contains(".staging-") {
            let removed_byte_count = cache_owned_directory_byte_count(&block_directory_path)?;
            remove_cache_owned_directory_or_confirm_absent(&block_directory_path)?;
            startup_cleanup_evidence
                .interrupted_transaction_recovery
                .record_block(removed_byte_count);
            continue;
        }
        let Some(block_hash_from_directory) =
            parse_persistent_prompt_cache_file_hash_from_path(&block_directory_path)
        else {
            let removed_byte_count = cache_owned_directory_byte_count(&block_directory_path)?;
            remove_cache_owned_directory_or_confirm_absent(&block_directory_path)?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_block(removed_byte_count);
            continue;
        };
        let block_manifest = match PersistentPromptCacheBlockManifest::read_from_block_directory(
            &block_directory_path,
            persistent_prompt_cache_model_contract,
        ) {
            Ok(block_manifest) => block_manifest,
            Err(_) => {
                let removed_byte_count = cache_owned_directory_byte_count(&block_directory_path)?;
                remove_cache_owned_directory_or_confirm_absent(&block_directory_path)?;
                startup_cleanup_evidence
                    .corrupt_current_format
                    .record_block(removed_byte_count);
                continue;
            }
        };
        if block_manifest.block_hash().ok() != Some(block_hash_from_directory) {
            let removed_byte_count = cache_owned_directory_byte_count(&block_directory_path)?;
            remove_cache_owned_directory_or_confirm_absent(&block_directory_path)?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_block(removed_byte_count);
            continue;
        }
        // Missing or invalid state remains `None` until topology reconciliation.
        // That later phase can distinguish a legal compacted parent boundary from
        // an incomplete leaf or a block missing required sequence state.
        let sequence_state_file = if block_manifest.has_sequence_state() {
            validate_block_state_file(
                &block_directory_path.join(SEQUENCE_STATE_FILE_NAME),
                PersistentPromptCacheFileKind::SequenceStateBlock,
                persistent_prompt_cache_model_contract,
            )?
        } else {
            None
        };
        let boundary_state_file = if block_manifest.has_boundary_state() {
            validate_block_state_file(
                &block_directory_path.join(BOUNDARY_STATE_FILE_NAME),
                PersistentPromptCacheFileKind::BoundaryStateSnapshot,
                persistent_prompt_cache_model_contract,
            )?
        } else {
            None
        };
        block_candidate_by_hash.insert(
            block_hash_from_directory,
            BlockDirectoryCandidate {
                block_directory_path,
                block_index: block_manifest.block_index(),
                parent_block_hash: block_manifest.parent_block_hash(),
                sequence_state_file,
                boundary_state_file,
            },
        );
    }
    reconcile_block_topology(
        block_candidate_by_hash,
        tracked_files,
        persistent_prompt_cache_model_contract,
        startup_cleanup_evidence,
    )?;
    Ok(())
}

struct BlockDirectoryCandidate {
    block_directory_path: PathBuf,
    block_index: u32,
    parent_block_hash: Option<[u8; 32]>,
    sequence_state_file: Option<TrackedPersistentPromptCacheFile>,
    boundary_state_file: Option<TrackedPersistentPromptCacheFile>,
}

fn validate_block_state_file(
    state_file_path: &Path,
    file_kind: PersistentPromptCacheFileKind,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<Option<TrackedPersistentPromptCacheFile>, PersistentPromptCacheDiskStoreError> {
    let state_file_size_bytes = match std::fs::symlink_metadata(state_file_path) {
        Ok(file_metadata) if file_metadata.is_file() => file_metadata.len(),
        Ok(_) => return Ok(None),
        Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(None);
        }
        Err(source) => {
            return Err(PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: state_file_path.to_path_buf(),
                source,
            });
        }
    };
    let state_file = open_without_following_symlinks(state_file_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::OpenBlockFile {
            block_file_path: state_file_path.to_path_buf(),
            source,
        }
    })?;
    if validate_current_file_header(
        file_kind,
        &state_file,
        state_file_path,
        persistent_prompt_cache_model_contract,
    )
    .is_err()
    {
        return Ok(None);
    }
    Ok(Some(TrackedPersistentPromptCacheFile {
        file_path: state_file_path.to_path_buf(),
        file_size_bytes: state_file_size_bytes,
    }))
}

fn reconcile_block_topology(
    mut block_candidate_by_hash: HashMap<[u8; 32], BlockDirectoryCandidate>,
    tracked_files: &mut PersistentPromptCacheDiskStoreIndex,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    startup_cleanup_evidence: &mut PersistentPromptCacheStartupCleanupEvidence,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    // Sequence ancestry is transitive. Removing one invalid parent can orphan a
    // child that looked valid in the previous pass, so iterate to a fixed point.
    loop {
        let invalid_block_hashes = block_candidate_by_hash
            .iter()
            .filter_map(|(block_hash, block_candidate)| {
                (!block_candidate_has_valid_sequence_ancestry(
                    block_candidate,
                    &block_candidate_by_hash,
                    persistent_prompt_cache_model_contract,
                ))
                .then_some(*block_hash)
            })
            .collect::<Vec<_>>();
        if invalid_block_hashes.is_empty() {
            break;
        }
        remove_invalid_block_candidates(
            &invalid_block_hashes,
            &mut block_candidate_by_hash,
            startup_cleanup_evidence,
        )?;
    }

    // Boundary retention is intentionally asymmetric: a non-checkpoint parent
    // may omit its boundary once a child exists, but leaves and checkpoints may not.
    let block_hashes_with_children = block_candidate_by_hash
        .values()
        .filter_map(|block_candidate| block_candidate.parent_block_hash)
        .collect::<HashSet<_>>();
    let invalid_boundary_block_hashes = block_candidate_by_hash
        .iter()
        .filter_map(|(block_hash, block_candidate)| {
            (!block_candidate_has_valid_boundary_topology(
                *block_hash,
                block_candidate,
                &block_hashes_with_children,
                persistent_prompt_cache_model_contract,
            ))
            .then_some(*block_hash)
        })
        .collect::<Vec<_>>();
    remove_invalid_block_candidates(
        &invalid_boundary_block_hashes,
        &mut block_candidate_by_hash,
        startup_cleanup_evidence,
    )?;
    // Boundary removals can create new sequence orphans. Prune those descendants
    // to a fixed point before exposing the surviving graph to lookup.
    loop {
        let orphan_block_hashes = block_candidate_by_hash
            .iter()
            .filter_map(|(block_hash, block_candidate)| {
                (block_candidate.block_index > 0
                    && block_candidate.parent_block_hash.is_none_or(|parent_hash| {
                        !block_candidate_by_hash.contains_key(&parent_hash)
                    }))
                .then_some(*block_hash)
            })
            .collect::<Vec<_>>();
        if orphan_block_hashes.is_empty() {
            break;
        }
        remove_invalid_block_candidates(
            &orphan_block_hashes,
            &mut block_candidate_by_hash,
            startup_cleanup_evidence,
        )?;
    }

    for (block_hash, block_candidate) in block_candidate_by_hash {
        tracked_files.insert_block(
            block_hash,
            TrackedPersistentPromptCacheBlock {
                block_directory_path: block_candidate.block_directory_path,
                block_index: block_candidate.block_index,
                parent_block_hash: block_candidate.parent_block_hash,
                sequence_state_file: block_candidate.sequence_state_file,
                boundary_state_file: block_candidate.boundary_state_file,
            },
        );
    }
    Ok(())
}

fn block_candidate_has_valid_sequence_ancestry(
    block_candidate: &BlockDirectoryCandidate,
    block_candidate_by_hash: &HashMap<[u8; 32], BlockDirectoryCandidate>,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> bool {
    if persistent_prompt_cache_model_contract.has_sequence_state()
        && block_candidate.sequence_state_file.is_none()
    {
        return false;
    }
    // A child edge is valid only when indices are consecutive. Content hashes do
    // not encode an independently inspectable ordinal, so the manifest supplies it.
    match (
        block_candidate.block_index,
        block_candidate.parent_block_hash,
    ) {
        (0, None) => true,
        (0, Some(_)) | (_, None) => false,
        (block_index, Some(parent_block_hash)) => block_candidate_by_hash
            .get(&parent_block_hash)
            .is_some_and(|parent_candidate| {
                parent_candidate.block_index.checked_add(1) == Some(block_index)
            }),
    }
}

fn block_candidate_has_valid_boundary_topology(
    block_hash: [u8; 32],
    block_candidate: &BlockDirectoryCandidate,
    block_hashes_with_children: &HashSet<[u8; 32]>,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> bool {
    if !persistent_prompt_cache_model_contract.has_boundary_state()
        || block_candidate.boundary_state_file.is_some()
    {
        return true;
    }
    // Only hybrid state can reconstruct a compacted parent's boundary from a
    // later boundary plus the complete sequence chain. Boundary-only models must
    // retain a snapshot at every indexed block.
    persistent_prompt_cache_model_contract.has_sequence_state()
        && block_hashes_with_children.contains(&block_hash)
        && !persistent_prompt_cache_boundary_is_common_prefix_checkpoint(
            block_candidate.block_index,
        )
}

fn remove_invalid_block_candidates(
    invalid_block_hashes: &[[u8; 32]],
    block_candidate_by_hash: &mut HashMap<[u8; 32], BlockDirectoryCandidate>,
    startup_cleanup_evidence: &mut PersistentPromptCacheStartupCleanupEvidence,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    for invalid_block_hash in invalid_block_hashes {
        if let Some(invalid_block_candidate) = block_candidate_by_hash.remove(invalid_block_hash) {
            let removed_byte_count =
                cache_owned_directory_byte_count(&invalid_block_candidate.block_directory_path)?;
            remove_cache_owned_directory_or_confirm_absent(
                &invalid_block_candidate.block_directory_path,
            )?;
            startup_cleanup_evidence
                .corrupt_current_format
                .record_block(removed_byte_count);
        }
    }
    Ok(())
}

fn cache_owned_file_byte_count(
    file_path: &Path,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    std::fs::symlink_metadata(file_path)
        .map(|file_metadata| file_metadata.len())
        .map_err(
            |source| PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: file_path.to_path_buf(),
                source,
            },
        )
}

fn cache_owned_directory_byte_count(
    directory_path: &Path,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    let mut pending_directories = vec![directory_path.to_path_buf()];
    let mut total_byte_count = 0_u64;
    while let Some(pending_directory) = pending_directories.pop() {
        let directory_entries = std::fs::read_dir(&pending_directory).map_err(|source| {
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
            } else {
                total_byte_count =
                    total_byte_count.saturating_add(cache_owned_file_byte_count(&entry_path)?);
            }
        }
    }
    Ok(total_byte_count)
}
