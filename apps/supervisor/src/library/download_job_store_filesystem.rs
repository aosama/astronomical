//! Symlink-rejecting path inspection helpers for durable Library transactions.

use std::fs::{self, File};
use std::path::{Path, PathBuf};

use super::DownloadJobStoreError;

pub(super) fn metadata_without_symlink(
    path: &Path,
) -> Result<Option<fs::Metadata>, DownloadJobStoreError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(DownloadJobStoreError::UnsafeFilesystemObject {
                path: path.to_path_buf(),
            })
        }
        Ok(metadata) => Ok(Some(metadata)),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(DownloadJobStoreError::InspectPath {
            path: path.to_path_buf(),
            source,
        }),
    }
}

pub(super) fn ensure_descendant_directory_or_absent(
    base_directory: &Path,
    directory_path: &Path,
) -> Result<(), DownloadJobStoreError> {
    let relative_directory = directory_path.strip_prefix(base_directory).map_err(|_| {
        DownloadJobStoreError::UnsafeFilesystemObject {
            path: directory_path.to_path_buf(),
        }
    })?;
    let mut current_path = base_directory.to_path_buf();
    for component in relative_directory.components() {
        current_path.push(component.as_os_str());
        let Some(metadata) = metadata_without_symlink(&current_path)? else {
            return Ok(());
        };
        if !metadata.is_dir() {
            return Err(DownloadJobStoreError::UnsafeFilesystemObject { path: current_path });
        }
    }
    Ok(())
}

pub(super) fn safely_join_existing_path(
    base_directory: &Path,
    relative_path: &Path,
) -> Result<PathBuf, DownloadJobStoreError> {
    let mut joined_path = base_directory.to_path_buf();
    for component in relative_path.components() {
        joined_path.push(component.as_os_str());
        let Some(metadata) = metadata_without_symlink(&joined_path)? else {
            return Ok(joined_path);
        };
        let is_final_component = joined_path == base_directory.join(relative_path);
        if !is_final_component && !metadata.is_dir() {
            return Err(DownloadJobStoreError::UnsafeFilesystemObject { path: joined_path });
        }
    }
    Ok(joined_path)
}

pub(super) fn remove_directory_if_empty(
    directory_path: &Path,
) -> Result<bool, DownloadJobStoreError> {
    match fs::remove_dir(directory_path) {
        Ok(()) => Ok(true),
        Err(source)
            if matches!(
                source.kind(),
                std::io::ErrorKind::DirectoryNotEmpty | std::io::ErrorKind::NotFound
            ) =>
        {
            Ok(false)
        }
        Err(source) => Err(DownloadJobStoreError::RemovePath {
            path: directory_path.to_path_buf(),
            source,
        }),
    }
}

pub(super) fn remove_file_if_present(file_path: &Path) -> Result<(), DownloadJobStoreError> {
    match fs::remove_file(file_path) {
        Ok(()) => Ok(()),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(DownloadJobStoreError::RemovePath {
            path: file_path.to_path_buf(),
            source,
        }),
    }
}

pub(super) fn synchronize_directory(directory_path: &Path) -> Result<(), DownloadJobStoreError> {
    File::open(directory_path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| DownloadJobStoreError::SynchronizeDirectory {
            path: directory_path.to_path_buf(),
            source,
        })
}
