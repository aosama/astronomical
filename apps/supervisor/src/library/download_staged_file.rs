//! Symlink-rejecting creation and append access for one hidden payload file.

use std::{
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

use super::{
    DownloadJobStoreError,
    download_job_store_filesystem::{metadata_without_symlink, synchronize_directory},
};

pub(super) fn open_staged_file_for_append(
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

pub(super) fn create_descendant_directories(
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
