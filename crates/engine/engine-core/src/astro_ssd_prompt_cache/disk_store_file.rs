//! Filesystem primitives shared by astro SSD prompt cache artifact owners.
//!
//! Decoder-state block files are written from host copies by
//! `host_safetensors_writer`, so the background publication writer never
//! touches MLX; projected visual embeddings still stream through MLX's
//! descriptor-backed writer. Publication uses no-follow opens and
//! temporary-name replacement.

use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use runtime_integration::MlxSafetensorsWriterError;

use model_primitives::performance_attribution::{PerformanceAttribution, PerformanceOperation};

use super::block_format::AstroSsdPromptCacheBlockHeader;
use super::block_format_error::AstroSsdPromptCacheBlockError;
use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::host_safetensors_writer::write_host_safetensors;
use super::host_state_tensor::AstroSsdPromptCacheHostTensors;
use super::model_contract::AstroSsdPromptCacheModelContract;

#[derive(Clone, Copy)]
pub enum AstroSsdPromptCacheFileKind {
    SequenceStateBlock,
    BoundaryStateSnapshot,
    VisualEmbedding,
}

pub(super) fn save_direct_safetensors_file_with_name(
    directory: &Path,
    file_name: &str,
    host_tensors: &AstroSsdPromptCacheHostTensors,
    block_token_count: usize,
    astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<StagedAstroSsdPromptCacheStateFile, AstroSsdPromptCacheDiskStoreError> {
    let (file_kind, serialization_operation) =
        state_file_kind_and_serialization_operation(file_name)?;
    let block_file_path = directory.join(file_name);
    let temporary_file_path = directory.join(format!("{file_name}.tmp"));
    remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(&temporary_file_path)?;
    let temporary_file = create_temporary_state_file(&temporary_file_path)?;
    // The tensors are host copies, so this write runs on whichever thread
    // publishes: the background writer for a prompt being read, which is
    // how a prompt's first token stops waiting for its blocks (measured
    // 2026-11-27, 15,000 tokens of Ornith 35B-A3B: ~420 ms of serialization
    // and directory syncs left the engine thread).
    //
    // The header records the block's own token count so partial tails
    // self-describe their geometry; tensor layout validation reads this same
    // header, keeping the file self-consistent.
    let prepared_metadata =
        PreparedSafetensorsMetadata::new(block_token_count, astro_ssd_prompt_cache_model_contract);
    let metadata_entries = prepared_metadata.entries();
    let written_byte_count = match performance_attribution.measure_operation(
        serialization_operation,
        |_performance_attribution| {
            write_host_safetensors(temporary_file, host_tensors, &metadata_entries)
        },
    ) {
        Ok(written_byte_count) => written_byte_count,
        Err(write_error) => {
            return remove_temporary_state_file_then_fail(&temporary_file_path, write_error);
        }
    };
    let actual_file_size_bytes = validate_written_temporary_state_file(
        &temporary_file_path,
        file_kind,
        written_byte_count,
        astro_ssd_prompt_cache_model_contract,
        performance_attribution,
    )?;
    rename_temporary_state_file_into_place(&temporary_file_path, &block_file_path)?;
    Ok(StagedAstroSsdPromptCacheStateFile {
        file_path: block_file_path,
        file_size_bytes: actual_file_size_bytes,
    })
}

/// Opens the temporary state file for a fresh write. `create_new` refuses a
/// leftover file and `O_NOFOLLOW` refuses a symlink planted at that name.
fn create_temporary_state_file(
    temporary_file_path: &Path,
) -> Result<File, AstroSsdPromptCacheDiskStoreError> {
    OpenOptions::new()
        .create_new(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(temporary_file_path)
        .map_err(|source| AstroSsdPromptCacheDiskStoreError::OpenTempFile {
            temp_file_path: temporary_file_path.to_path_buf(),
            source,
        })
}

/// Removes the temporary file and then reports `failure`. A removal that
/// itself fails is reported instead, because it leaves a file behind that the
/// original failure does not explain.
fn remove_temporary_state_file_then_fail<Outcome>(
    temporary_file_path: &Path,
    failure: AstroSsdPromptCacheDiskStoreError,
) -> Result<Outcome, AstroSsdPromptCacheDiskStoreError> {
    remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(temporary_file_path)?;
    Err(failure)
}

/// Checks the temporary file after the write and returns its size:
/// the size the filesystem sees must match what the writer reported, and the
/// header and tensor layout must validate against the model contract for the
/// file's kind. Every failure removes the temporary file before it propagates.
fn validate_written_temporary_state_file(
    temporary_file_path: &Path,
    file_kind: AstroSsdPromptCacheFileKind,
    written_byte_count: u64,
    astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<u64, AstroSsdPromptCacheDiskStoreError> {
    // Verify both native accounting and the safetensors contract before rename.
    // A successful native status alone is not sufficient evidence of a complete
    // file when descriptor I/O callbacks can fail independently.
    let actual_file_size_bytes =
        match confirm_written_file_size_bytes(temporary_file_path, written_byte_count) {
            Ok(actual_file_size_bytes) => actual_file_size_bytes,
            Err(metadata_error) => {
                return remove_temporary_state_file_then_fail(temporary_file_path, metadata_error);
            }
        };
    let temporary_file = match open_without_following_symlinks(temporary_file_path) {
        Ok(temporary_file) => temporary_file,
        Err(source) => {
            return remove_temporary_state_file_then_fail(
                temporary_file_path,
                AstroSsdPromptCacheDiskStoreError::OpenBlockFile {
                    block_file_path: temporary_file_path.to_path_buf(),
                    source,
                },
            );
        }
    };
    if let Err(source) = performance_attribution.measure_operation(
        PerformanceOperation::AstroSsdPromptCachePublicationValidation,
        |_performance_attribution| {
            validate_current_file_header(
                file_kind,
                &temporary_file,
                temporary_file_path,
                astro_ssd_prompt_cache_model_contract,
            )
        },
    ) {
        return remove_temporary_state_file_then_fail(
            temporary_file_path,
            AstroSsdPromptCacheDiskStoreError::ValidateBlock {
                block_file_path: temporary_file_path.to_path_buf(),
                source,
            },
        );
    }
    Ok(actual_file_size_bytes)
}

/// Moves the validated temporary file to its final name. The temporary file
/// is removed when the rename fails.
fn rename_temporary_state_file_into_place(
    temporary_file_path: &Path,
    block_file_path: &Path,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    // This rename publishes only within a private block staging directory. The
    // enclosing transaction performs the durability sync and final block rename.
    if let Err(source) = fs::rename(temporary_file_path, block_file_path) {
        let rename_error = AstroSsdPromptCacheDiskStoreError::RenameTempFile {
            temp_file_path: temporary_file_path.to_path_buf(),
            block_file_path: block_file_path.to_path_buf(),
            source,
        };
        return remove_temporary_state_file_then_fail(temporary_file_path, rename_error);
    }
    Ok(())
}

pub(super) struct StagedAstroSsdPromptCacheStateFile {
    pub(super) file_path: PathBuf,
    pub(super) file_size_bytes: u64,
}

/// Accepts only the two contract-owned state-file names, mapping each to its
/// state kind and its serialization attribution operation. Allowing an
/// arbitrary caller name would bypass state-kind validation and broaden
/// cleanup authority.
fn state_file_kind_and_serialization_operation(
    file_name: &str,
) -> Result<(AstroSsdPromptCacheFileKind, PerformanceOperation), AstroSsdPromptCacheDiskStoreError>
{
    match file_name {
        super::block_manifest::SEQUENCE_STATE_FILE_NAME => Ok((
            AstroSsdPromptCacheFileKind::SequenceStateBlock,
            PerformanceOperation::AstroSsdPromptCacheKvBlockSerialization,
        )),
        super::block_manifest::BOUNDARY_STATE_FILE_NAME => Ok((
            AstroSsdPromptCacheFileKind::BoundaryStateSnapshot,
            PerformanceOperation::AstroSsdPromptCacheRecurrentSnapshotSerialization,
        )),
        _ => Err(AstroSsdPromptCacheDiskStoreError::InvalidStateFileName {
            file_name: file_name.to_owned(),
        }),
    }
}

/// The safetensors header metadata a state file carries: the format version,
/// the block's own token count (so partial tails self-describe their
/// geometry), and the model contract's storage fingerprint (so a layout
/// change can never be silently mixed with old blocks). Owns the two derived
/// strings so the borrowed entries live as long as one serialization call.
struct PreparedSafetensorsMetadata {
    block_token_count_metadata: String,
    storage_contract_fingerprint_metadata: String,
}

impl PreparedSafetensorsMetadata {
    fn new(
        block_token_count: usize,
        astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
    ) -> Self {
        PreparedSafetensorsMetadata {
            block_token_count_metadata: block_token_count.to_string(),
            storage_contract_fingerprint_metadata: astro_ssd_prompt_cache_model_contract
                .storage_contract_fingerprint_hex(),
        }
    }

    fn entries(&self) -> [(&str, &str); 3] {
        [
            (
                "format_version",
                super::block_format::ASTRO_SSD_PROMPT_CACHE_FORMAT_VERSION,
            ),
            (
                "block_token_count",
                self.block_token_count_metadata.as_str(),
            ),
            (
                "storage_contract_fingerprint",
                self.storage_contract_fingerprint_metadata.as_str(),
            ),
        ]
    }
}

/// Confirms the bytes the native writer reported match the file the
/// filesystem sees. Native accounting and descriptor I/O can disagree when a
/// callback fails independently, so both numbers must be checked before the
/// file is allowed to progress toward publication.
fn confirm_written_file_size_bytes(
    temporary_file_path: &Path,
    written_byte_count: u64,
) -> Result<u64, AstroSsdPromptCacheDiskStoreError> {
    let actual_file_size_bytes = read_file_size_bytes(temporary_file_path)?;
    if actual_file_size_bytes != written_byte_count {
        return Err(AstroSsdPromptCacheDiskStoreError::WrittenFileSizeMismatch {
            file_path: temporary_file_path.to_path_buf(),
            reported_size_bytes: written_byte_count,
            actual_size_bytes: actual_file_size_bytes,
        });
    }
    Ok(actual_file_size_bytes)
}

pub fn map_safetensors_writer_error(
    writer_error: MlxSafetensorsWriterError,
) -> AstroSsdPromptCacheDiskStoreError {
    match writer_error {
        MlxSafetensorsWriterError::DescriptorIo { source } => {
            AstroSsdPromptCacheDiskStoreError::WriteSafetensorsDescriptor { source }
        }
        MlxSafetensorsWriterError::Native { source } => {
            AstroSsdPromptCacheDiskStoreError::SaveSafetensors { source }
        }
    }
}

pub fn synchronize_directory(
    directory_path: &Path,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    // fsyncing file contents does not guarantee the directory entry survives a
    // crash. Callers use this after rename to persist the name-to-inode update.
    let directory = File::open(directory_path).map_err(|source| {
        AstroSsdPromptCacheDiskStoreError::OpenBlockFile {
            block_file_path: directory_path.to_path_buf(),
            source,
        }
    })?;
    directory.sync_all().map_err(|source| {
        AstroSsdPromptCacheDiskStoreError::SynchronizeAstroSsdPromptCacheDirectory {
            astro_ssd_prompt_cache_directory: directory_path.to_path_buf(),
            source,
        }
    })
}

pub fn remove_astro_ssd_prompt_cache_owned_directory_or_confirm_absent(
    astro_ssd_prompt_cache_directory_path: &Path,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    fs::remove_dir_all(astro_ssd_prompt_cache_directory_path).or_else(|removal_error| {
        if removal_error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(
                AstroSsdPromptCacheDiskStoreError::RemoveAstroSsdPromptCacheFile {
                    astro_ssd_prompt_cache_file_path: astro_ssd_prompt_cache_directory_path
                        .to_path_buf(),
                    source: removal_error,
                },
            )
        }
    })
}

/// Returns the parsed header's block token count for state-bearing file kinds,
/// or `None` for kinds without a block-token axis. Callers that load by block
/// key use the count to fail closed when a file's stored token count disagrees
/// with the key that addressed it (partial tails make this reachable).
pub(super) fn validate_current_file_header(
    file_kind: AstroSsdPromptCacheFileKind,
    file: &File,
    file_path: &Path,
    astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
) -> Result<Option<usize>, AstroSsdPromptCacheBlockError> {
    // Dispatch by semantic ownership, not filename. Each reader validates the
    // exact model-bound metadata and tensor layout required by that artifact.
    match file_kind {
        AstroSsdPromptCacheFileKind::SequenceStateBlock => {
            let block_header = AstroSsdPromptCacheBlockHeader::read_kv_block_from_file(
                file,
                file_path,
                astro_ssd_prompt_cache_model_contract,
            )?;
            Ok(Some(block_header.block_token_count()))
        }
        AstroSsdPromptCacheFileKind::BoundaryStateSnapshot => {
            let block_header = AstroSsdPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
                file,
                file_path,
                astro_ssd_prompt_cache_model_contract,
            )?;
            Ok(Some(block_header.block_token_count()))
        }
        AstroSsdPromptCacheFileKind::VisualEmbedding => Ok(None),
    }
}

pub(super) fn expected_tensor_names(
    file_kind: AstroSsdPromptCacheFileKind,
    astro_ssd_prompt_cache_model_contract: &AstroSsdPromptCacheModelContract,
) -> Vec<String> {
    let mut tensor_names = Vec::new();
    match file_kind {
        AstroSsdPromptCacheFileKind::SequenceStateBlock
        | AstroSsdPromptCacheFileKind::BoundaryStateSnapshot => {
            let expected_tensor_layouts = match file_kind {
                AstroSsdPromptCacheFileKind::SequenceStateBlock => {
                    astro_ssd_prompt_cache_model_contract
                        .astro_decoder_state_cache_layout()
                        .sequence_tensor_layouts()
                }
                AstroSsdPromptCacheFileKind::BoundaryStateSnapshot => {
                    astro_ssd_prompt_cache_model_contract
                        .astro_decoder_state_cache_layout()
                        .boundary_tensor_layouts()
                }
                AstroSsdPromptCacheFileKind::VisualEmbedding => Vec::new(),
            };
            tensor_names.extend(
                expected_tensor_layouts
                    .into_iter()
                    .map(|expected_tensor_layout| expected_tensor_layout.persistent_tensor_name()),
            );
        }
        AstroSsdPromptCacheFileKind::VisualEmbedding => {}
    }
    tensor_names
}

pub fn read_file_size_bytes(file_path: &Path) -> Result<u64, AstroSsdPromptCacheDiskStoreError> {
    fs::symlink_metadata(file_path)
        .map(|metadata| metadata.len())
        .map_err(
            |source| AstroSsdPromptCacheDiskStoreError::ReadBlockMetadata {
                block_file_path: file_path.to_path_buf(),
                source,
            },
        )
}

/// The hash an astro SSD prompt cache file or block directory is named after.
pub(super) fn parse_astro_ssd_prompt_cache_file_hash_from_path(
    entry_path: &Path,
) -> Option<[u8; 32]> {
    lowercase_hex::decode_lowercase_hex::<32>(entry_path.file_stem()?.to_str()?)
}

pub fn open_without_following_symlinks(path: &Path) -> Result<File, std::io::Error> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

/// Removes one astro SSD prompt cache file, treating `NotFound` as already-absent (cleanup
/// complete) and mapping every other failure to `RemoveAstroSsdPromptCacheFile`.
///
/// `NotFound` is success only where absence proves cleanup is complete: stale
/// temp removal, oversize/invalid-format rollback, eviction of an already-gone
/// file, and corrupt-load deletion after a concurrent remover. Callers that
/// must observe a present file (load `open`) do not use this helper.
pub fn remove_astro_ssd_prompt_cache_owned_file_or_confirm_absent(
    astro_ssd_prompt_cache_file_path: &Path,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    fs::remove_file(astro_ssd_prompt_cache_file_path).or_else(|removal_error| {
        if removal_error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(
                AstroSsdPromptCacheDiskStoreError::RemoveAstroSsdPromptCacheFile {
                    astro_ssd_prompt_cache_file_path: astro_ssd_prompt_cache_file_path
                        .to_path_buf(),
                    source: removal_error,
                },
            )
        }
    })
}

#[cfg(test)]
#[path = "disk_store_file_tests.rs"]
mod disk_store_file_tests;
