//! Atomic persistence, restart recovery, and destructive cleanup for one download job.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, MutexGuard,
    atomic::{AtomicU64, Ordering},
};

use super::download_job_store_filesystem::{
    ensure_descendant_directory_or_absent, metadata_without_symlink, remove_directory_if_empty,
    remove_file_if_present, safely_join_existing_path, synchronize_directory,
};
use super::download_job_store_lock::acquire_existing_store_transaction_lock;
use super::{DownloadJob, DownloadJobState, DownloadJobStoreError};

const DOWNLOAD_JOB_FILE_NAME: &str = ".download-job.json";
const MAXIMUM_DOWNLOAD_JOB_BYTES: u64 = 8_000_000;
const MAXIMUM_TEMPORARY_FILE_ATTEMPTS: u64 = 100;
static NEXT_TEMPORARY_JOB_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Filesystem owner for the one durable job beneath an instance models directory.
#[derive(Clone, Debug)]
pub struct DownloadJobStore {
    pub(super) models_directory: PathBuf,
    operation_lock: Arc<Mutex<()>>,
}

impl DownloadJobStore {
    #[must_use]
    pub fn new(models_directory: PathBuf) -> Self {
        Self {
            models_directory,
            operation_lock: Arc::new(Mutex::new(())),
        }
    }

    #[must_use]
    pub fn job_file_path(&self) -> PathBuf {
        self.models_directory.join(DOWNLOAD_JOB_FILE_NAME)
    }

    #[must_use]
    pub fn models_directory(&self) -> &Path {
        &self.models_directory
    }

    /// Loads and validates the durable document without changing its state or progress.
    pub fn load(&self) -> Result<Option<DownloadJob>, DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        self.load_unlocked()
    }

    pub(super) fn load_unlocked(&self) -> Result<Option<DownloadJob>, DownloadJobStoreError> {
        match metadata_without_symlink(&self.models_directory)? {
            None => return Ok(None),
            Some(metadata) if metadata.is_dir() => {}
            Some(_) => {
                return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                    path: self.models_directory.clone(),
                });
            }
        }
        let job_file_path = self.job_file_path();
        let Some(job_file_metadata) = metadata_without_symlink(&job_file_path)? else {
            return Ok(None);
        };
        if !job_file_metadata.is_file() {
            return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                path: job_file_path,
            });
        }
        if job_file_metadata.len() > MAXIMUM_DOWNLOAD_JOB_BYTES {
            return Err(DownloadJobStoreError::JobDocumentTooLarge);
        }
        let job_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&job_file_path)
            .map_err(|source| DownloadJobStoreError::OpenJob {
                path: job_file_path.clone(),
                source,
            })?;
        let read_capacity = usize::try_from(job_file_metadata.len())
            .unwrap_or(8_000_000)
            .min(8_000_000);
        let mut job_bytes = Vec::with_capacity(read_capacity);
        job_file
            .take(MAXIMUM_DOWNLOAD_JOB_BYTES.saturating_add(1))
            .read_to_end(&mut job_bytes)
            .map_err(|source| DownloadJobStoreError::ReadJob {
                path: job_file_path.clone(),
                source,
            })?;
        if job_bytes.len() as u64 > MAXIMUM_DOWNLOAD_JOB_BYTES {
            return Err(DownloadJobStoreError::JobDocumentTooLarge);
        }
        let job_json =
            std::str::from_utf8(&job_bytes).map_err(DownloadJobStoreError::InvalidUtf8)?;
        Ok(Some(DownloadJob::parse_json(job_json)?))
    }

    /// Creates the first job without replacing an existing active or resumable job.
    pub fn create(&self, download_job: &DownloadJob) -> Result<(), DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        download_job.validate()?;
        let serialized_job = download_job
            .to_json_bytes()
            .map_err(DownloadJobStoreError::SerializeJob)?;
        if serialized_job.len() as u64 > MAXIMUM_DOWNLOAD_JOB_BYTES {
            return Err(DownloadJobStoreError::JobDocumentTooLarge);
        }
        self.prepare_models_directory()?;
        let _transaction_lock = acquire_existing_store_transaction_lock(&self.models_directory)?
            .ok_or(DownloadJobStoreError::OperationLockUnavailable)?;
        let temporary_job_file_path = self.write_temporary_job(&serialized_job)?;
        let job_file_path = self.job_file_path();
        match fs::hard_link(&temporary_job_file_path, &job_file_path) {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                let _cleanup_outcome = fs::remove_file(&temporary_job_file_path);
                return Err(DownloadJobStoreError::JobAlreadyExists);
            }
            Err(source) => {
                let _cleanup_outcome = fs::remove_file(&temporary_job_file_path);
                return Err(DownloadJobStoreError::CreateJob {
                    path: job_file_path,
                    source,
                });
            }
        }
        // The hard link is the commit point. A stale hidden temporary name is recoverable;
        // reporting the committed job as absent would make callers start conflicting work.
        if let Err(cleanup_error) = fs::remove_file(&temporary_job_file_path) {
            tracing::warn!(
                temporary_job_file = %temporary_job_file_path.display(),
                error = %cleanup_error,
                "download job was committed but its temporary link could not be removed"
            );
        }
        synchronize_directory(&self.models_directory)
    }

    pub(super) fn save_unlocked(
        &self,
        download_job: &DownloadJob,
    ) -> Result<(), DownloadJobStoreError> {
        download_job.validate()?;
        let serialized_job = download_job
            .to_json_bytes()
            .map_err(DownloadJobStoreError::SerializeJob)?;
        if serialized_job.len() as u64 > MAXIMUM_DOWNLOAD_JOB_BYTES {
            return Err(DownloadJobStoreError::JobDocumentTooLarge);
        }
        self.prepare_models_directory()?;
        let temporary_job_file_path = self.write_temporary_job(&serialized_job)?;
        let job_file_path = self.job_file_path();
        if let Err(source) = fs::rename(&temporary_job_file_path, &job_file_path) {
            let _cleanup_outcome = fs::remove_file(&temporary_job_file_path);
            return Err(DownloadJobStoreError::ReplaceJob {
                path: job_file_path,
                source,
            });
        }
        synchronize_directory(&self.models_directory)
    }

    /// Atomically replaces the current job while preserving its immutable model identity.
    pub fn replace_current(
        &self,
        replacement_job: &DownloadJob,
    ) -> Result<(), DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let _transaction_lock = acquire_existing_store_transaction_lock(&self.models_directory)?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        let current_job = self
            .load_unlocked()?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        if current_job.state() == DownloadJobState::Publishing {
            return Err(DownloadJobStoreError::PublishingRecoveryRequired);
        }
        if current_job.huggingface_id() != replacement_job.huggingface_id()
            || current_job.revision() != replacement_job.revision()
        {
            return Err(DownloadJobStoreError::JobIdentityMismatch);
        }
        let resets_invalid_file_for_retry = current_job.state() == DownloadJobState::Verifying
            && replacement_job.state() == DownloadJobState::Failed
            && replacement_job.error_code()
                == Some(super::DownloadJobPublicErrorCode::ChecksumMismatch);
        if replacement_job.updated_at_unix_millis() < current_job.updated_at_unix_millis()
            || (replacement_job.bytes_completed() < current_job.bytes_completed()
                && !resets_invalid_file_for_retry)
            || (current_job.has_exact_manifest() && !replacement_job.has_exact_manifest())
            || !current_job
                .state()
                .can_replace_with(replacement_job.state())
        {
            return Err(DownloadJobStoreError::InvalidJobReplacement);
        }
        self.save_unlocked(replacement_job)
    }

    /// Reconciles progress from regular staged files and pauses interrupted work durably.
    pub fn load_recovering_interrupted_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<Option<DownloadJob>, DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let Some(_transaction_lock) =
            acquire_existing_store_transaction_lock(&self.models_directory)?
        else {
            return Ok(None);
        };
        let Some(mut download_job) = self.load_unlocked()? else {
            return Ok(None);
        };
        if download_job.state() == DownloadJobState::Publishing {
            return Err(DownloadJobStoreError::PublishingRecoveryRequired);
        }
        let bytes_on_disk_by_file = self.read_staged_file_lengths(&download_job)?;
        download_job.reconcile_file_progress(
            &bytes_on_disk_by_file,
            updated_at_unix_millis,
            true,
        )?;
        self.save_unlocked(&download_job)?;
        Ok(Some(download_job))
    }

    /// Reconciles and durably pauses the current job before acknowledging the caller.
    pub fn pause_current_job(
        &self,
        updated_at_unix_millis: u64,
    ) -> Result<Option<DownloadJob>, DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let Some(_transaction_lock) =
            acquire_existing_store_transaction_lock(&self.models_directory)?
        else {
            return Ok(None);
        };
        let Some(mut download_job) = self.load_unlocked()? else {
            return Ok(None);
        };
        if download_job.state() == DownloadJobState::Publishing {
            return Err(DownloadJobStoreError::PublishingRecoveryRequired);
        }
        let bytes_on_disk_by_file = self.read_staged_file_lengths(&download_job)?;
        download_job.reconcile_file_progress(
            &bytes_on_disk_by_file,
            updated_at_unix_millis,
            false,
        )?;
        download_job.force_paused(updated_at_unix_millis);
        download_job.validate()?;
        self.save_unlocked(&download_job)?;
        Ok(Some(download_job))
    }

    /// Removes hidden staging and the job document. Absence is idempotent success.
    pub fn cancel_current_job(&self) -> Result<bool, DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let Some(_transaction_lock) =
            acquire_existing_store_transaction_lock(&self.models_directory)?
        else {
            return Ok(false);
        };
        let Some(download_job) = self.load_unlocked()? else {
            return Ok(false);
        };
        if download_job.state() == DownloadJobState::Publishing {
            return Err(DownloadJobStoreError::PublishingRecoveryRequired);
        }
        let staging_directory = download_job.staging_directory(&self.models_directory);
        self.remove_staging_directory_if_present(&staging_directory)?;
        remove_file_if_present(&self.job_file_path())?;
        synchronize_directory(&self.models_directory)?;
        Ok(true)
    }

    fn prepare_models_directory(&self) -> Result<(), DownloadJobStoreError> {
        match metadata_without_symlink(&self.models_directory)? {
            Some(metadata) if metadata.is_dir() => Ok(()),
            Some(_) => Err(DownloadJobStoreError::UnsafeFilesystemObject {
                path: self.models_directory.clone(),
            }),
            None => {
                match fs::create_dir(&self.models_directory) {
                    Ok(()) => {}
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {
                        let concurrent_metadata = metadata_without_symlink(&self.models_directory)?;
                        if !concurrent_metadata.is_some_and(|metadata| metadata.is_dir()) {
                            return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                                path: self.models_directory.clone(),
                            });
                        }
                    }
                    Err(source) => {
                        return Err(DownloadJobStoreError::CreateDirectory {
                            path: self.models_directory.clone(),
                            source,
                        });
                    }
                }
                let parent_directory = self.models_directory.parent().ok_or_else(|| {
                    DownloadJobStoreError::CreateDirectory {
                        path: self.models_directory.clone(),
                        source: std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "models directory has no parent",
                        ),
                    }
                })?;
                synchronize_directory(parent_directory)
            }
        }
    }

    fn write_temporary_job(&self, serialized_job: &[u8]) -> Result<PathBuf, DownloadJobStoreError> {
        for attempt_number in 0..MAXIMUM_TEMPORARY_FILE_ATTEMPTS {
            let sequence_number = NEXT_TEMPORARY_JOB_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let temporary_job_file_path = self.models_directory.join(format!(
                ".download-job.json.tmp.{}.{}",
                std::process::id(),
                sequence_number.saturating_add(attempt_number),
            ));
            let mut temporary_job_file = match OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&temporary_job_file_path)
            {
                Ok(temporary_job_file) => temporary_job_file,
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(DownloadJobStoreError::WriteTemporaryJob {
                        path: temporary_job_file_path,
                        source,
                    });
                }
            };
            if let Err(source) = temporary_job_file
                .write_all(serialized_job)
                .and_then(|()| temporary_job_file.sync_all())
            {
                let _cleanup_outcome = fs::remove_file(&temporary_job_file_path);
                return Err(DownloadJobStoreError::WriteTemporaryJob {
                    path: temporary_job_file_path,
                    source,
                });
            }
            return Ok(temporary_job_file_path);
        }
        Err(DownloadJobStoreError::WriteTemporaryJob {
            path: self.models_directory.clone(),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique temporary download job file",
            ),
        })
    }

    fn read_staged_file_lengths(
        &self,
        download_job: &DownloadJob,
    ) -> Result<Vec<u64>, DownloadJobStoreError> {
        let staging_directory = download_job.staging_directory(&self.models_directory);
        ensure_descendant_directory_or_absent(&self.models_directory, &staging_directory)?;
        download_job
            .files()
            .iter()
            .map(|download_file| {
                let staged_file_path = safely_join_existing_path(
                    &staging_directory,
                    Path::new(download_file.relative_path()),
                )?;
                let Some(staged_file_metadata) = metadata_without_symlink(&staged_file_path)?
                else {
                    return Ok(0);
                };
                if !staged_file_metadata.is_file() {
                    return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                        path: staged_file_path,
                    });
                }
                if staged_file_metadata.len() > download_file.expected_bytes() {
                    return Err(DownloadJobStoreError::StagedFileTooLarge {
                        path: staged_file_path,
                    });
                }
                let staged_file = OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NOFOLLOW)
                    .open(&staged_file_path)
                    .map_err(|source| DownloadJobStoreError::OpenStagedFile {
                        path: staged_file_path.clone(),
                        source,
                    })?;
                staged_file.sync_all().map_err(|source| {
                    DownloadJobStoreError::SynchronizeStagedFile {
                        path: staged_file_path.clone(),
                        source,
                    }
                })?;
                self.synchronize_staging_ancestors(&staged_file_path)?;
                let synchronized_file_size = staged_file
                    .metadata()
                    .map_err(|source| DownloadJobStoreError::InspectPath {
                        path: staged_file_path.clone(),
                        source,
                    })?
                    .len();
                if synchronized_file_size > download_file.expected_bytes() {
                    return Err(DownloadJobStoreError::StagedFileTooLarge {
                        path: staged_file_path,
                    });
                }
                Ok(synchronized_file_size)
            })
            .collect()
    }

    pub(super) fn remove_staging_directory_if_present(
        &self,
        staging_directory: &Path,
    ) -> Result<(), DownloadJobStoreError> {
        ensure_descendant_directory_or_absent(&self.models_directory, staging_directory)?;
        let staging_directory_exists = metadata_without_symlink(staging_directory)?.is_some();
        if staging_directory_exists {
            fs::remove_dir_all(staging_directory).map_err(|source| {
                DownloadJobStoreError::RemovePath {
                    path: staging_directory.to_path_buf(),
                    source,
                }
            })?;
            self.synchronize_removed_staging_ancestors(staging_directory)?;
        }
        Ok(())
    }

    fn synchronize_removed_staging_ancestors(
        &self,
        staging_directory: &Path,
    ) -> Result<(), DownloadJobStoreError> {
        let Some(organization_directory) = staging_directory.parent() else {
            return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                path: staging_directory.to_path_buf(),
            });
        };
        synchronize_directory(organization_directory)?;
        let incomplete_directory = organization_directory.parent().ok_or_else(|| {
            DownloadJobStoreError::UnsafeFilesystemObject {
                path: organization_directory.to_path_buf(),
            }
        })?;
        if remove_directory_if_empty(organization_directory)? {
            synchronize_directory(incomplete_directory)?;
        }
        if remove_directory_if_empty(incomplete_directory)? {
            synchronize_directory(&self.models_directory)?;
        }
        Ok(())
    }

    fn synchronize_staging_ancestors(
        &self,
        staged_file_path: &Path,
    ) -> Result<(), DownloadJobStoreError> {
        let mut current_directory = staged_file_path.parent().ok_or_else(|| {
            DownloadJobStoreError::UnsafeFilesystemObject {
                path: staged_file_path.to_path_buf(),
            }
        })?;
        loop {
            if !current_directory.starts_with(&self.models_directory) {
                return Err(DownloadJobStoreError::UnsafeFilesystemObject {
                    path: current_directory.to_path_buf(),
                });
            }
            synchronize_directory(current_directory)?;
            if current_directory == self.models_directory {
                return Ok(());
            }
            current_directory = current_directory.parent().ok_or_else(|| {
                DownloadJobStoreError::UnsafeFilesystemObject {
                    path: staged_file_path.to_path_buf(),
                }
            })?;
        }
    }

    pub(super) fn lock_operations(&self) -> Result<MutexGuard<'_, ()>, DownloadJobStoreError> {
        self.operation_lock
            .lock()
            .map_err(|_| DownloadJobStoreError::OperationLockUnavailable)
    }
}
