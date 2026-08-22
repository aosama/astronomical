//! Locked publication transactions kept separate from ordinary durable-job persistence.

use std::{fs, path::PathBuf};

use super::{
    DownloadJob, DownloadJobState, DownloadJobStore, DownloadJobStoreError,
    DownloadPublicationRefresh,
    download_job_store_filesystem::{
        ensure_descendant_directory_or_absent, metadata_without_symlink, remove_file_if_present,
        synchronize_directory,
    },
    download_job_store_lock::acquire_existing_store_transaction_lock,
    download_staged_file::create_descendant_directories,
};

impl DownloadJobStore {
    /// Persists publication intent only while the verified final destination is absent.
    pub(crate) fn replace_current_for_publication(
        &self,
        replacement_job: &DownloadJob,
    ) -> Result<(), DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let _transaction_lock = acquire_existing_store_transaction_lock(&self.models_directory)?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        let current_job = self
            .load_unlocked()?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        if current_job.state() != DownloadJobState::Verifying
            || replacement_job.state() != DownloadJobState::Publishing
            || current_job.huggingface_id() != replacement_job.huggingface_id()
            || current_job.revision() != replacement_job.revision()
            || replacement_job.updated_at_unix_millis() < current_job.updated_at_unix_millis()
            || replacement_job.bytes_completed() != current_job.bytes_completed()
        {
            return Err(DownloadJobStoreError::InvalidJobReplacement);
        }
        let published_directory = self.published_directory(replacement_job);
        ensure_descendant_directory_or_absent(&self.models_directory, &published_directory)?;
        if metadata_without_symlink(&published_directory)?.is_some() {
            return Err(DownloadJobStoreError::PublishedModelAlreadyExists);
        }
        let staging_directory = replacement_job.staging_directory(&self.models_directory);
        ensure_descendant_directory_or_absent(&self.models_directory, &staging_directory)?;
        if !metadata_without_symlink(&staging_directory)?.is_some_and(|metadata| metadata.is_dir())
        {
            return Err(DownloadJobStoreError::InconsistentPublicationState);
        }
        self.save_unlocked(replacement_job)
    }

    /// Atomically publishes verified staging and completes only after discovery refresh succeeds.
    pub(crate) fn publish_current_job(
        &self,
        discovery_refresh: &dyn DownloadPublicationRefresh,
    ) -> Result<PathBuf, DownloadJobStoreError> {
        let _operation_guard = self.lock_operations()?;
        let _transaction_lock = acquire_existing_store_transaction_lock(&self.models_directory)?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        let download_job = self
            .load_unlocked()?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        if download_job.state() != DownloadJobState::Publishing {
            return Err(DownloadJobStoreError::PublishingRecoveryRequired);
        }
        let staging_directory = download_job.staging_directory(&self.models_directory);
        ensure_descendant_directory_or_absent(&self.models_directory, &staging_directory)?;
        let published_directory = self.published_directory(&download_job);
        ensure_descendant_directory_or_absent(&self.models_directory, &published_directory)?;
        let organization_directory = published_directory
            .parent()
            .ok_or(DownloadJobStoreError::InconsistentPublicationState)?;
        match (
            metadata_without_symlink(&staging_directory)?,
            metadata_without_symlink(&published_directory)?,
        ) {
            (Some(staging), None) if staging.is_dir() => {
                create_descendant_directories(&self.models_directory, organization_directory)?;
                fs::rename(&staging_directory, &published_directory).map_err(|source| {
                    DownloadJobStoreError::PublishModel {
                        path: published_directory.clone(),
                        source,
                    }
                })?;
                synchronize_directory(organization_directory)?;
                synchronize_directory(&self.models_directory)?;
            }
            (None, Some(published)) if published.is_dir() => {}
            _ => return Err(DownloadJobStoreError::InconsistentPublicationState),
        }
        discovery_refresh
            .refresh(&published_directory)
            .map_err(|source| DownloadJobStoreError::DiscoveryRefresh { source })?;
        remove_file_if_present(&self.job_file_path())?;
        synchronize_directory(&self.models_directory)?;
        Ok(published_directory)
    }

    fn published_directory(&self, download_job: &DownloadJob) -> PathBuf {
        let mut identity_components = download_job.huggingface_id().split('/');
        let organization = identity_components.next().unwrap_or_default();
        let model_name = identity_components.next().unwrap_or_default();
        self.models_directory.join(organization).join(model_name)
    }
}
