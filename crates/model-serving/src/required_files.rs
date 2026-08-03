use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};

use crate::validated_artifact::{ValidatedRequiredFile, validated_file_identity};
use crate::{ArtifactValidationError, RequiredFileProfile};

const CAPTURE_BUFFER_SIZE_BYTES: usize = 64 * 1024;
const CONFIG_FILE_NAME: &str = "config.json";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

/// Maximum file size for which we capture bytes during validation so runtime
/// loading never reopens mutable paths. The Qwen3.5-MoE tokenizer is about 20 MB;
/// this bound covers it and config.json with fixed headroom while preventing
/// unbounded trust-boundary reads.
const MAXIMUM_CAPTURED_FILE_SIZE_BYTES: u64 = 32 * 1024 * 1024;

pub(crate) fn validate_required_files(
    model_directory: &Path,
    required_file_profiles: &[RequiredFileProfile],
) -> Result<HashMap<String, ValidatedRequiredFile>, ArtifactValidationError> {
    let mut required_file_paths = HashMap::with_capacity(required_file_profiles.len());
    for required_file_profile in required_file_profiles {
        let required_file_path = validate_required_file(model_directory, required_file_profile)?;
        required_file_paths.insert(required_file_profile.file_name.clone(), required_file_path);
    }

    Ok(required_file_paths)
}

pub(crate) fn validate_required_file(
    model_directory: &Path,
    required_file_profile: &RequiredFileProfile,
) -> Result<ValidatedRequiredFile, ArtifactValidationError> {
    validate_required_file_relative_path(&required_file_profile.file_name)?;
    let required_file_path = model_directory.join(&required_file_profile.file_name);
    let file_metadata = fs::symlink_metadata(&required_file_path).map_err(|source| {
        ArtifactValidationError::InspectRequiredFile {
            file_name: required_file_profile.file_name.clone(),
            source,
        }
    })?;

    // Ordinary model directories remain symlink-free. Hugging Face snapshots are
    // the one safe exception because their files intentionally point into the
    // sibling immutable `blobs/` directory.
    let required_file_open_path = if file_metadata.file_type().is_symlink() {
        resolve_hugging_face_snapshot_blob_path(
            model_directory,
            &required_file_path,
            &required_file_profile.file_name,
        )?
        .ok_or_else(|| ArtifactValidationError::RequiredFileIsSymlink {
            file_name: required_file_profile.file_name.clone(),
        })?
    } else if file_metadata.file_type().is_file() {
        required_file_path
    } else {
        return Err(ArtifactValidationError::RequiredFileIsNotRegular {
            file_name: required_file_profile.file_name.clone(),
        });
    };

    // Open the resolved blob itself with O_NOFOLLOW. If the final path changes
    // to a symlink after validation, the kernel rejects it instead of following it.
    let file =
        open_read_only_without_following_symlinks(&required_file_open_path).map_err(|source| {
            ArtifactValidationError::InspectRequiredFile {
                file_name: required_file_profile.file_name.clone(),
                source,
            }
        })?;
    let file_metadata =
        file.metadata()
            .map_err(|source| ArtifactValidationError::InspectRequiredFile {
                file_name: required_file_profile.file_name.clone(),
                source,
            })?;
    if !file_metadata.file_type().is_file() {
        return Err(ArtifactValidationError::RequiredFileIsNotRegular {
            file_name: required_file_profile.file_name.clone(),
        });
    }

    let actual_size_bytes = file_metadata.len();
    if required_file_profile.size_bytes != 0
        && actual_size_bytes != required_file_profile.size_bytes
    {
        return Err(ArtifactValidationError::RequiredFileSizeMismatch {
            file_name: required_file_profile.file_name.clone(),
            expected_size_bytes: required_file_profile.size_bytes,
            actual_size_bytes,
        });
    }

    let should_capture_bytes = matches!(
        required_file_profile.file_name.as_str(),
        CONFIG_FILE_NAME | TOKENIZER_FILE_NAME
    );
    if should_capture_bytes && actual_size_bytes > MAXIMUM_CAPTURED_FILE_SIZE_BYTES {
        return Err(ArtifactValidationError::CapturedRequiredFileTooLarge {
            file_name: required_file_profile.file_name.clone(),
            actual_size_bytes,
            maximum_size_bytes: MAXIMUM_CAPTURED_FILE_SIZE_BYTES,
        });
    }
    let captured_bytes = if should_capture_bytes {
        Some(capture_file(&file, actual_size_bytes).map_err(|source| {
            ArtifactValidationError::ReadRequiredFileForCapture {
                file_name: required_file_profile.file_name.clone(),
                source,
            }
        })?)
    } else {
        None
    };
    let final_file_metadata =
        file.metadata().map_err(
            |source| ArtifactValidationError::ReadRequiredFileForCapture {
                file_name: required_file_profile.file_name.clone(),
                source,
            },
        )?;
    if validated_file_identity(&final_file_metadata) != validated_file_identity(&file_metadata) {
        return Err(ArtifactValidationError::ValidatedFileIdentityChanged {
            file_name: required_file_profile.file_name.clone(),
        });
    }

    Ok(ValidatedRequiredFile::new(
        file,
        validated_file_identity(&file_metadata),
        required_file_profile.file_name.clone(),
        actual_size_bytes,
        captured_bytes,
    ))
}

/// Test seam for required-file path validation without exposing retained internal metadata.
#[doc(hidden)]
pub fn validate_required_file_for_tests(
    model_directory: &Path,
    required_file_profile: &RequiredFileProfile,
) -> Result<crate::ValidatedWeightsFile, ArtifactValidationError> {
    validate_required_file(model_directory, required_file_profile)?.into_validated_weights_file()
}

fn validate_required_file_relative_path(
    required_file_name: &str,
) -> Result<(), ArtifactValidationError> {
    let required_file_path = Path::new(required_file_name);
    // Profiles describe one file inside the model directory. Reject absolute,
    // parent, current-directory, and platform-prefix components before joining.
    if required_file_name.is_empty()
        || required_file_path.is_absolute()
        || required_file_path
            .components()
            .any(|path_component| !matches!(path_component, Component::Normal(_)))
    {
        return Err(ArtifactValidationError::InvalidProfileFileName {
            file_name: required_file_name.to_owned(),
        });
    }
    Ok(())
}

fn resolve_hugging_face_snapshot_blob_path(
    model_directory: &Path,
    required_file_path: &Path,
    required_file_name: &str,
) -> Result<Option<PathBuf>, ArtifactValidationError> {
    let Some(model_cache_directory) = hugging_face_model_cache_directory(model_directory) else {
        return Ok(None);
    };

    // Canonicalize both sides before comparison. Path text such as `../` or a
    // nested symlink cannot make an outside target appear to be inside `blobs/`.
    let blob_directory = model_cache_directory.join("blobs");
    let blob_directory_metadata = fs::symlink_metadata(&blob_directory).map_err(|source| {
        ArtifactValidationError::InspectRequiredFile {
            file_name: required_file_name.to_owned(),
            source,
        }
    })?;
    if blob_directory_metadata.file_type().is_symlink()
        || !blob_directory_metadata.file_type().is_dir()
    {
        return Err(ArtifactValidationError::RequiredFileIsSymlink {
            file_name: required_file_name.to_owned(),
        });
    }
    let canonical_blob_directory = fs::canonicalize(&blob_directory).map_err(|source| {
        ArtifactValidationError::InspectRequiredFile {
            file_name: required_file_name.to_owned(),
            source,
        }
    })?;
    let canonical_blob_path = fs::canonicalize(required_file_path).map_err(|source| {
        ArtifactValidationError::InspectRequiredFile {
            file_name: required_file_name.to_owned(),
            source,
        }
    })?;
    if !canonical_blob_path.starts_with(&canonical_blob_directory) {
        return Err(
            ArtifactValidationError::HuggingFaceSnapshotSymlinkEscapesBlobDirectory {
                file_name: required_file_name.to_owned(),
                resolved_target_path: canonical_blob_path,
                expected_blob_directory: canonical_blob_directory,
            },
        );
    }
    Ok(Some(canonical_blob_path))
}

pub(crate) fn hugging_face_snapshot_model_id(model_directory: &Path) -> Option<String> {
    let model_cache_directory = hugging_face_model_cache_directory(model_directory)?;
    let cache_directory_name = model_cache_directory.file_name()?.to_str()?;
    let provider_qualified_model_id =
        astronomical_config::decode_huggingface_cache_directory_name(cache_directory_name)?;
    Some(
        provider_qualified_model_id
            .split_once('/')
            .map_or(provider_qualified_model_id.clone(), |(_, model_id)| {
                model_id.to_owned()
            }),
    )
}

fn hugging_face_model_cache_directory(model_directory: &Path) -> Option<&Path> {
    let snapshots_directory = model_directory.parent()?;
    if snapshots_directory
        .file_name()
        .and_then(|name| name.to_str())
        != Some("snapshots")
    {
        return None;
    }
    let model_cache_directory = snapshots_directory.parent()?;
    let cache_directory_name = model_cache_directory.file_name()?.to_str()?;
    astronomical_config::decode_huggingface_cache_directory_name(cache_directory_name)?;
    Some(model_cache_directory)
}

fn open_read_only_without_following_symlinks(file_path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(file_path)
}

fn capture_file(file: &File, size_bytes: u64) -> io::Result<Vec<u8>> {
    let mut capture_buffer = [0_u8; CAPTURE_BUFFER_SIZE_BYTES];
    let captured_size = usize::try_from(size_bytes).map_err(io::Error::other)?;
    let mut captured_bytes = Vec::new();
    captured_bytes
        .try_reserve_exact(captured_size)
        .map_err(io::Error::other)?;
    let mut offset_bytes = 0_u64;

    while offset_bytes < size_bytes {
        let remaining_bytes = size_bytes - offset_bytes;
        let requested_bytes =
            usize::try_from(remaining_bytes.min(CAPTURE_BUFFER_SIZE_BYTES as u64))
                .map_err(io::Error::other)?;
        let bytes_read = file.read_at(&mut capture_buffer[..requested_bytes], offset_bytes)?;
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "validated file ended before its inspected size",
            ));
        }
        captured_bytes.extend_from_slice(&capture_buffer[..bytes_read]);
        offset_bytes = offset_bytes
            .checked_add(bytes_read as u64)
            .ok_or_else(|| io::Error::other("validated file read offset overflowed"))?;
    }

    Ok(captured_bytes)
}
