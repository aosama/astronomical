//! Symlink-rejecting creation and append access for hidden payload files, including the
//! zero-byte files that the transfer window never schedules.

use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::library::download::job::store::filesystem::{
    metadata_without_symlink, synchronize_directory,
};
use crate::library::{DownloadJob, DownloadJobStoreError};

pub(in crate::library::download) fn open_staged_file_for_append(
    models_directory: &Path,
    staged_file_path: &Path,
    resume_offset_bytes: u64,
) -> Result<File, DownloadJobStoreError> {
    let staged_parent =
        staged_file_path
            .parent()
            .ok_or_else(|| DownloadJobStoreError::UnsafeFilesystemObject {
                path: staged_file_path.to_path_buf(),
            })?;
    create_descendant_directories(models_directory, staged_parent)?;
    if let Some(existing_metadata) = metadata_without_symlink(staged_file_path)?
        && !existing_metadata.is_file()
    {
        return Err(DownloadJobStoreError::UnsafeFilesystemObject {
            path: staged_file_path.to_path_buf(),
        });
    }
    let mut staged_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(staged_file_path)
        .map_err(|source| DownloadJobStoreError::OpenStagedFile {
            path: staged_file_path.to_path_buf(),
            source,
        })?;
    let actual_file_bytes = staged_file
        .metadata()
        .map_err(|source| DownloadJobStoreError::InspectPath {
            path: staged_file_path.to_path_buf(),
            source,
        })?
        .len();
    if actual_file_bytes != resume_offset_bytes {
        return Err(DownloadJobStoreError::StagedFileProgressMismatch {
            path: staged_file_path.to_path_buf(),
        });
    }
    staged_file
        .seek(SeekFrom::Start(resume_offset_bytes))
        .map_err(|source| DownloadJobStoreError::OpenStagedFile {
            path: staged_file_path.to_path_buf(),
            source,
        })?;
    synchronize_directory(staged_parent)?;
    Ok(staged_file)
}

/// How creating the empty files ended.
#[derive(Debug, PartialEq, Eq)]
pub(in crate::library::download) enum EmptyFileCreation {
    Created,
    /// A pause or cancel was requested first; some empty files may be missing, and a resume
    /// creates them, since creation is idempotent.
    Interrupted,
}

/// Creates the staged copy of every zero-byte file in `download_job`. The
/// transfer window only schedules files with bytes left to fetch, so an empty
/// file (an `__init__.py`, say) would otherwise never be created, and
/// verification would refuse the download as missing it on every attempt
/// (issue #1896). Creating it is the whole download of such a file.
///
/// The loop runs on a blocking thread, which a cancel cannot abort. It therefore checks
/// `pause_requested` before each file, so a cancel that removes the staging folder is not
/// followed by this loop re-creating files inside it.
pub(in crate::library::download) fn create_empty_staged_files(
    models_directory: &Path,
    download_job: &DownloadJob,
    pause_requested: &AtomicBool,
) -> Result<EmptyFileCreation, DownloadJobStoreError> {
    let staging_directory = download_job.staging_directory(models_directory);
    for download_file in download_job.files() {
        if download_file.expected_bytes() != 0 {
            continue;
        }
        if pause_requested.load(Ordering::Acquire) {
            return Ok(EmptyFileCreation::Interrupted);
        }
        let staged_file_path = staging_directory.join(download_file.relative_path());
        open_staged_file_for_append(models_directory, &staged_file_path, 0)?;
    }
    Ok(EmptyFileCreation::Created)
}

pub(in crate::library::download) fn create_descendant_directories(
    base_directory: &Path,
    directory_path: &Path,
) -> Result<(), DownloadJobStoreError> {
    let relative_directory = directory_path.strip_prefix(base_directory).map_err(|_| {
        DownloadJobStoreError::UnsafeFilesystemObject {
            path: directory_path.to_path_buf(),
        }
    })?;
    let mut current_directory = PathBuf::from(base_directory);
    for path_component in relative_directory.components() {
        let parent_directory = current_directory.clone();
        current_directory.push(path_component.as_os_str());
        match metadata_without_symlink(&current_directory)? {
            Some(metadata) if metadata.is_dir() => {}
            Some(_) => {
                return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                    path: current_directory,
                });
            }
            None => {
                match fs::create_dir(&current_directory) {
                    Ok(()) => {}
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(source) => {
                        return Err(DownloadJobStoreError::CreateDirectory {
                            path: current_directory,
                            source,
                        });
                    }
                }
                let created_metadata = metadata_without_symlink(&current_directory)?;
                if !created_metadata.is_some_and(|metadata| metadata.is_dir()) {
                    return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                        path: current_directory,
                    });
                }
                synchronize_directory(&parent_directory)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "staged_file_tests.rs"]
mod staged_file_tests;
