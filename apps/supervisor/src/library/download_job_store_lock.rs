//! Advisory transaction lock shared by independent job-store instances and daemon processes.

use std::{fs::File, os::unix::fs::OpenOptionsExt, path::Path};

use super::{DownloadJobStoreError, download_job_store_filesystem::metadata_without_symlink};

const DOWNLOAD_JOB_LOCK_FILE_NAME: &str = ".download-job.lock";

/// Open file ownership keeps the advisory lock held for one complete filesystem transaction.
pub(super) struct DownloadJobStoreTransactionLock {
    _lock_file: File,
}

impl DownloadJobStoreTransactionLock {
    pub(super) fn acquire(models_directory: &Path) -> Result<Self, DownloadJobStoreError> {
        let lock_file_path = models_directory.join(DOWNLOAD_JOB_LOCK_FILE_NAME);
        let lock_file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&lock_file_path)
            .map_err(|source| DownloadJobStoreError::AcquireTransactionLock {
                path: lock_file_path.clone(),
                source,
            })?;
        lock_file
            .lock()
            .map_err(|source| DownloadJobStoreError::AcquireTransactionLock {
                path: lock_file_path,
                source,
            })?;
        Ok(Self {
            _lock_file: lock_file,
        })
    }
}

pub(super) fn acquire_existing_store_transaction_lock(
    models_directory: &Path,
) -> Result<Option<DownloadJobStoreTransactionLock>, DownloadJobStoreError> {
    let Some(models_metadata) = metadata_without_symlink(models_directory)? else {
        return Ok(None);
    };
    if !models_metadata.is_dir() {
        return Err(DownloadJobStoreError::UnsafeFilesystemObject {
            path: models_directory.to_path_buf(),
        });
    }
    DownloadJobStoreTransactionLock::acquire(models_directory).map(Some)
}
