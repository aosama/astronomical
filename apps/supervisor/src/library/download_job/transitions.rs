//! Transfer-only state transitions kept separate from durable document validation.

use super::{DownloadJob, DownloadJobError, DownloadJobPublicErrorCode, DownloadJobState};

impl DownloadJob {
    pub(crate) fn mark_downloading(
        &mut self,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if !self.has_exact_manifest()
            || !matches!(
                self.state,
                DownloadJobState::Paused | DownloadJobState::Failed
            )
        {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.state = DownloadJobState::Downloading;
        self.error_code = None;
        self.current_file_relative_path = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn select_download_file(
        &mut self,
        relative_path: &str,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state != DownloadJobState::Downloading
            || !self.files.iter().any(|download_file| {
                download_file.relative_path == relative_path
                    && download_file.bytes_on_disk < download_file.expected_bytes
            })
        {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.current_file_relative_path = Some(relative_path.to_owned());
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn record_file_progress(
        &mut self,
        relative_path: &str,
        bytes_on_disk: u64,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state != DownloadJobState::Downloading
            || self.current_file_relative_path.as_deref() != Some(relative_path)
        {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        let file_index = self
            .files
            .iter()
            .position(|download_file| download_file.relative_path == relative_path)
            .ok_or(DownloadJobError::InvalidStateTransition)?;
        let download_file = &mut self.files[file_index];
        if bytes_on_disk < download_file.bytes_on_disk
            || bytes_on_disk > download_file.expected_bytes
        {
            return Err(DownloadJobError::InvalidFileProgress { file_index });
        }
        let progress_increment = bytes_on_disk - download_file.bytes_on_disk;
        download_file.bytes_on_disk = bytes_on_disk;
        self.bytes_completed = self
            .bytes_completed
            .checked_add(progress_increment)
            .ok_or(DownloadJobError::InconsistentByteTotals)?;
        if bytes_on_disk == download_file.expected_bytes {
            self.current_file_relative_path = None;
        }
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn mark_verifying(
        &mut self,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state != DownloadJobState::Downloading || self.bytes_completed != self.bytes_total {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.state = DownloadJobState::Verifying;
        self.current_file_relative_path = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn mark_publishing(
        &mut self,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state != DownloadJobState::Verifying {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.state = DownloadJobState::Publishing;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn mark_failed(
        &mut self,
        error_code: DownloadJobPublicErrorCode,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state == DownloadJobState::Publishing {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.state = DownloadJobState::Failed;
        self.error_code = Some(error_code);
        self.current_file_relative_path = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn mark_checksum_failed_for_retry(
        &mut self,
        relative_path: &str,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.state != DownloadJobState::Verifying {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        let file_index = self
            .files
            .iter()
            .position(|download_file| download_file.relative_path == relative_path)
            .ok_or(DownloadJobError::InvalidStateTransition)?;
        let invalid_file_bytes = self.files[file_index].bytes_on_disk;
        self.files[file_index].bytes_on_disk = 0;
        self.bytes_completed = self
            .bytes_completed
            .checked_sub(invalid_file_bytes)
            .ok_or(DownloadJobError::InconsistentByteTotals)?;
        self.state = DownloadJobState::Failed;
        self.error_code = Some(DownloadJobPublicErrorCode::ChecksumMismatch);
        self.current_file_relative_path = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }
}
