//! Filesystem primitives shared by prompt-cache artifact owners.
//!
//! Format-11 decoder state takes the direct writer path: MLX serializes tensors
//! through one retained file descriptor without constructing a second complete
//! Rust byte payload. Publication uses no-follow opens and temporary-name replacement.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxSafetensorsWriterError};

use crate::{PerformanceAttribution, PerformanceOperation};

use super::block_format::PersistentPromptCacheBlockHeader;
use super::block_format_error::PersistentPromptCacheBlockError;
use super::disk_store_error::PersistentPromptCacheDiskStoreError;
use super::model_contract::PersistentPromptCacheModelContract;

#[derive(Clone, Copy)]
pub(crate) enum PersistentPromptCacheFileKind {
    SequenceStateBlock,
    BoundaryStateSnapshot,
    VisualEmbedding,
}

pub(super) fn save_direct_safetensors_file_with_name(
    runtime: &MlxRuntime,
    directory: &Path,
    file_name: &str,
    tensors: &HashMap<String, MlxArray>,
    block_token_count: usize,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StagedPersistentPromptCacheStateFile, PersistentPromptCacheDiskStoreError> {
    // Accept only the two contract-owned names. Allowing an arbitrary caller
    // name would bypass state-kind validation and broaden cleanup authority.
    let (file_kind, serialization_operation) = match file_name {
        super::block_manifest::SEQUENCE_STATE_FILE_NAME => (
            PersistentPromptCacheFileKind::SequenceStateBlock,
            PerformanceOperation::PersistentPromptCacheKvBlockSerialization,
        ),
        super::block_manifest::BOUNDARY_STATE_FILE_NAME => (
            PersistentPromptCacheFileKind::BoundaryStateSnapshot,
            PerformanceOperation::PersistentPromptCacheRecurrentSnapshotSerialization,
        ),
        _ => {
            return Err(PersistentPromptCacheDiskStoreError::InvalidStateFileName {
                file_name: file_name.to_owned(),
            });
        }
    };
    let block_file_path = directory.join(file_name);
    let temporary_file_path = directory.join(format!("{file_name}.tmp"));
    remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
    let temporary_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temporary_file_path)
        .map_err(|source| PersistentPromptCacheDiskStoreError::OpenTempFile {
            temp_file_path: temporary_file_path.clone(),
            source,
        })?;
    // These references remain valid for the entire synchronous native call.
    // The retained descriptor streams each MLX materialization to disk; there is
    // deliberately no queue, detached thread, or serialized block-sized buffer.
    let named_arrays: Vec<(&str, &MlxArray)> = tensors
        .iter()
        .map(|(tensor_name, tensor)| (tensor_name.as_str(), tensor))
        .collect();
    // The header records the block's own token count so partial tails
    // self-describe their geometry; tensor layout validation reads this same
    // header, keeping the file self-consistent.
    let block_token_count_metadata = block_token_count.to_string();
    let storage_contract_fingerprint_metadata =
        persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex();
    let metadata_entries: [(&str, &str); 3] = [
        (
            "format_version",
            super::block_format::PERSISTENT_PROMPT_CACHE_FORMAT_VERSION,
        ),
        ("block_token_count", block_token_count_metadata.as_str()),
        (
            "storage_contract_fingerprint",
            storage_contract_fingerprint_metadata.as_str(),
        ),
    ];
    let write_outcome = match performance_attribution.measure_operation(
        serialization_operation,
        |_performance_attribution| {
            runtime.save_safetensors(temporary_file, &named_arrays, &metadata_entries)
        },
    ) {
        Ok(write_outcome) => write_outcome,
        Err(source) => {
            remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
            return Err(map_safetensors_writer_error(source));
        }
    };
    // Verify both native accounting and the safetensors contract before rename.
    // A successful native status alone is not sufficient evidence of a complete
    // file when descriptor I/O callbacks can fail independently.
    let actual_file_size_bytes = match read_file_size_bytes(&temporary_file_path) {
        Ok(actual_file_size_bytes) => actual_file_size_bytes,
        Err(metadata_error) => {
            remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
            return Err(metadata_error);
        }
    };
    if actual_file_size_bytes != write_outcome.written_byte_count() {
        let size_mismatch_error = PersistentPromptCacheDiskStoreError::WrittenFileSizeMismatch {
            file_path: temporary_file_path.clone(),
            reported_size_bytes: write_outcome.written_byte_count(),
            actual_size_bytes: actual_file_size_bytes,
        };
        remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
        return Err(size_mismatch_error);
    }
    let temporary_file = match open_without_following_symlinks(&temporary_file_path) {
        Ok(temporary_file) => temporary_file,
        Err(source) => {
            remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
            return Err(PersistentPromptCacheDiskStoreError::OpenBlockFile {
                block_file_path: temporary_file_path,
                source,
            });
        }
    };
    if let Err(source) = performance_attribution.measure_operation(
        PerformanceOperation::PersistentPromptCachePublicationValidation,
        |_performance_attribution| {
            validate_current_file_header(
                file_kind,
                &temporary_file,
                &temporary_file_path,
                persistent_prompt_cache_model_contract,
            )
        },
    ) {
        remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
        return Err(PersistentPromptCacheDiskStoreError::ValidateBlock {
            block_file_path: temporary_file_path,
            source,
        });
    }
    // This rename publishes only within a private block staging directory. The
    // enclosing transaction performs the durability sync and final block rename.
    if let Err(source) = fs::rename(&temporary_file_path, &block_file_path) {
        let rename_error = PersistentPromptCacheDiskStoreError::RenameTempFile {
            temp_file_path: temporary_file_path.clone(),
            block_file_path: block_file_path.clone(),
            source,
        };
        remove_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
        return Err(rename_error);
    }
    Ok(StagedPersistentPromptCacheStateFile {
        file_path: block_file_path,
        file_size_bytes: actual_file_size_bytes,
    })
}

pub(super) struct StagedPersistentPromptCacheStateFile {
    pub(super) file_path: PathBuf,
    pub(super) file_size_bytes: u64,
}

pub(crate) fn map_safetensors_writer_error(
    writer_error: MlxSafetensorsWriterError,
) -> PersistentPromptCacheDiskStoreError {
    match writer_error {
        MlxSafetensorsWriterError::DescriptorIo { source } => {
            PersistentPromptCacheDiskStoreError::WriteSafetensorsDescriptor { source }
        }
        MlxSafetensorsWriterError::Native { source } => {
            PersistentPromptCacheDiskStoreError::SaveSafetensors { source }
        }
    }
}

pub(crate) fn synchronize_directory(
    directory_path: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    // fsyncing file contents does not guarantee the directory entry survives a
    // crash. Callers use this after rename to persist the name-to-inode update.
    let directory = File::open(directory_path).map_err(|source| {
        PersistentPromptCacheDiskStoreError::OpenBlockFile {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    directory.sync_all().map_err(|source| {
        PersistentPromptCacheDiskStoreError::SynchronizePromptCacheDirectory {
            persistent_prompt_cache_directory: directory_path.to_path_buf(),
            source,
        }
    })
}

pub(crate) fn remove_cache_owned_directory_or_confirm_absent(
    persistent_prompt_cache_directory_path: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    fs::remove_dir_all(persistent_prompt_cache_directory_path).or_else(|removal_error| {
        if removal_error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
                persistent_prompt_cache_file_path: persistent_prompt_cache_directory_path
                    .to_path_buf(),
                source: removal_error,
            })
        }
    })
}

/// Returns the parsed header's block token count for state-bearing file kinds,
/// or `None` for kinds without a block-token axis. Callers that load by block
/// key use the count to fail closed when a file's stored token count disagrees
/// with the key that addressed it (partial tails make this reachable).
pub(super) fn validate_current_file_header(
    file_kind: PersistentPromptCacheFileKind,
    file: &File,
    file_path: &Path,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Result<Option<usize>, PersistentPromptCacheBlockError> {
    // Dispatch by semantic ownership, not filename. Each reader validates the
    // exact model-bound metadata and tensor layout required by that artifact.
    match file_kind {
        PersistentPromptCacheFileKind::SequenceStateBlock => {
            let block_header = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
                file,
                file_path,
                persistent_prompt_cache_model_contract,
            )?;
            Ok(Some(block_header.block_token_count()))
        }
        PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
            let block_header = PersistentPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
                file,
                file_path,
                persistent_prompt_cache_model_contract,
            )?;
            Ok(Some(block_header.block_token_count()))
        }
        PersistentPromptCacheFileKind::VisualEmbedding => return Ok(None),
    }
}

pub(super) fn expected_tensor_names(
    file_kind: PersistentPromptCacheFileKind,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
) -> Vec<String> {
    let mut tensor_names = Vec::new();
    match file_kind {
        PersistentPromptCacheFileKind::SequenceStateBlock
        | PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
            let expected_tensor_layouts = match file_kind {
                PersistentPromptCacheFileKind::SequenceStateBlock => {
                    persistent_prompt_cache_model_contract
                        .decoder_cache_layout()
                        .sequence_tensor_layouts()
                }
                PersistentPromptCacheFileKind::BoundaryStateSnapshot => {
                    persistent_prompt_cache_model_contract
                        .decoder_cache_layout()
                        .boundary_tensor_layouts()
                }
                PersistentPromptCacheFileKind::VisualEmbedding => Vec::new(),
            };
            tensor_names.extend(
                expected_tensor_layouts
                    .into_iter()
                    .map(|expected_tensor_layout| expected_tensor_layout.persistent_tensor_name()),
            );
        }
        PersistentPromptCacheFileKind::VisualEmbedding => {}
    }
    tensor_names
}

pub(crate) fn read_file_size_bytes(
    file_path: &Path,
) -> Result<u64, PersistentPromptCacheDiskStoreError> {
    fs::symlink_metadata(file_path)
        .map(|metadata| metadata.len())
        .map_err(
            |source| PersistentPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: file_path.to_path_buf(),
                source,
            },
        )
}

pub(super) fn parse_persistent_prompt_cache_file_hash_from_path(
    entry_path: &Path,
) -> Option<[u8; 32]> {
    let file_stem = entry_path.file_stem()?.to_str()?;
    if file_stem.len() != 64 {
        return None;
    }
    let mut persistent_prompt_cache_file_hash = [0_u8; 32];
    for (byte_index, persistent_prompt_cache_file_hash_byte) in
        persistent_prompt_cache_file_hash.iter_mut().enumerate()
    {
        let hex_pair = &file_stem[byte_index * 2..byte_index * 2 + 2];
        *persistent_prompt_cache_file_hash_byte = u8::from_str_radix(hex_pair, 16).ok()?;
    }
    Some(persistent_prompt_cache_file_hash)
}

pub(crate) fn open_without_following_symlinks(path: &Path) -> Result<File, std::io::Error> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

/// Removes one cache-owned file, treating `NotFound` as already-absent (cleanup
/// complete) and mapping every other failure to `RemovePromptCacheFile`.
///
/// `NotFound` is success only where absence proves cleanup is complete: stale
/// temp removal, oversize/invalid-format rollback, eviction of an already-gone
/// file, and corrupt-load deletion after a concurrent remover. Callers that
/// must observe a present file (load `open`) do not use this helper.
pub(crate) fn remove_cache_owned_file_or_confirm_absent(
    persistent_prompt_cache_file_path: &Path,
) -> Result<(), PersistentPromptCacheDiskStoreError> {
    fs::remove_file(persistent_prompt_cache_file_path).or_else(|removal_error| {
        if removal_error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(PersistentPromptCacheDiskStoreError::RemovePromptCacheFile {
                persistent_prompt_cache_file_path: persistent_prompt_cache_file_path.to_path_buf(),
                source: removal_error,
            })
        }
    })
}

pub(crate) fn hex_encode(block_hash_bytes: [u8; 32]) -> String {
    block_hash_bytes
        .iter()
        .map(|block_hash_byte| format!("{block_hash_byte:02x}"))
        .collect()
}
