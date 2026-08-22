//! Strict durable state for one instance-owned model download.

mod path_validation;
mod transitions;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    DownloadFileDigest, HuggingFaceManifest,
    download_catalog::{is_valid_huggingface_id, is_valid_immutable_revision},
};
use path_validation::{has_path_hierarchy_conflict, is_safe_relative_path};

pub(crate) const MAXIMUM_DOWNLOAD_JOB_BYTES: usize = 8_000_000;
const MAXIMUM_DOWNLOAD_JOB_FILE_COUNT: usize = 65_536;
const MAXIMUM_JAVASCRIPT_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Durable state of one transfer from preflight through atomic publication.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadJob {
    huggingface_id: String,
    revision: String,
    state: DownloadJobState,
    bytes_completed: u64,
    bytes_total: u64,
    current_file_relative_path: Option<String>,
    files: Vec<DownloadJobFile>,
    error_code: Option<DownloadJobPublicErrorCode>,
    #[serde(rename = "updated_at")]
    updated_at_unix_millis: u64,
}

/// One manifest file and its resumable local progress.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadJobFile {
    relative_path: String,
    expected_bytes: u64,
    expected_digest: DownloadFileDigest,
    bytes_on_disk: u64,
}

/// Persisted job states. Successful cancellation has no durable state because it deletes the job.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadJobState {
    CheckingDisk,
    FetchingManifest,
    Downloading,
    Paused,
    Verifying,
    Publishing,
    Failed,
}

/// Stable public failures that Observatory can present without local path disclosure.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadJobPublicErrorCode {
    LibraryBusy,
    CatalogEntryNotFound,
    ModelNotPublic,
    InsufficientDisk,
    DownloadGated,
    ChecksumMismatch,
    DownloadFailed,
    ModelAlreadyPresent,
}

/// Syntax or semantic validation failure in durable job metadata.
#[derive(Debug, Error)]
pub enum DownloadJobError {
    #[error("download job exceeds the 8000000-byte metadata limit")]
    DocumentTooLarge,
    #[error("download job is not valid JSON: {0}")]
    Parse(#[source] serde_json::Error),
    #[error("download job has an invalid Hugging Face identity")]
    InvalidHuggingFaceId,
    #[error("download job has an invalid immutable revision")]
    InvalidRevision,
    #[error("download job exceeds the 65536-file metadata limit")]
    TooManyFiles,
    #[error("download job file {file_index} has an unsafe relative path")]
    UnsafeRelativePath { file_index: usize },
    #[error("download job contains a duplicate or case-colliding file path")]
    DuplicateFilePath,
    #[error("download job contains a file and descendant path conflict")]
    FilePathHierarchyConflict,
    #[error("download job file {file_index} has an invalid provider digest")]
    InvalidDigest { file_index: usize },
    #[error("download job file {file_index} has invalid byte progress")]
    InvalidFileProgress { file_index: usize },
    #[error("download job byte totals are inconsistent")]
    InconsistentByteTotals,
    #[error("download job current file is not in the manifest or is not downloading")]
    InvalidCurrentFile,
    #[error("download job failure code does not match its state")]
    InvalidErrorState,
    #[error("download job state transition is invalid")]
    InvalidStateTransition,
}

impl DownloadJob {
    /// Creates the durable admission state before any Hub metadata request is allowed.
    pub fn new_checking_disk(
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        catalog_approximate_bytes: u64,
        updated_at_unix_millis: u64,
    ) -> Result<Self, DownloadJobError> {
        let download_job = Self {
            huggingface_id: huggingface_id.into(),
            revision: revision.into(),
            state: DownloadJobState::CheckingDisk,
            bytes_completed: 0,
            bytes_total: catalog_approximate_bytes,
            current_file_relative_path: None,
            files: Vec::new(),
            error_code: None,
            updated_at_unix_millis,
        };
        download_job.validate()?;
        Ok(download_job)
    }

    /// Converts one validated exact manifest into paused durable transfer state.
    pub fn from_manifest(
        manifest: &HuggingFaceManifest,
        updated_at_unix_millis: u64,
    ) -> Result<Self, DownloadJobError> {
        let files = manifest
            .files()
            .iter()
            .map(|manifest_file| DownloadJobFile {
                relative_path: manifest_file.relative_path().to_owned(),
                expected_bytes: manifest_file.expected_bytes(),
                expected_digest: manifest_file.digest().clone(),
                bytes_on_disk: 0,
            })
            .collect();
        let download_job = Self {
            huggingface_id: manifest.repository_id().to_owned(),
            revision: manifest.revision().to_owned(),
            state: DownloadJobState::Paused,
            bytes_completed: 0,
            bytes_total: manifest.total_bytes(),
            current_file_relative_path: None,
            files,
            error_code: None,
            updated_at_unix_millis,
        };
        download_job.validate()?;
        Ok(download_job)
    }

    /// Marks the persisted boundary immediately before immutable Hub metadata retrieval.
    pub fn mark_fetching_manifest(
        &mut self,
        updated_at_unix_millis: u64,
    ) -> Result<(), DownloadJobError> {
        if self.has_exact_manifest()
            || !matches!(
                self.state,
                DownloadJobState::CheckingDisk
                    | DownloadJobState::Paused
                    | DownloadJobState::Failed
            )
        {
            return Err(DownloadJobError::InvalidStateTransition);
        }
        self.state = DownloadJobState::FetchingManifest;
        self.error_code = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    /// Parses one complete bounded job document and validates all derived paths and totals.
    pub fn parse_json(job_json: &str) -> Result<Self, DownloadJobError> {
        if job_json.len() > MAXIMUM_DOWNLOAD_JOB_BYTES {
            return Err(DownloadJobError::DocumentTooLarge);
        }
        let download_job: Self = serde_json::from_str(job_json).map_err(DownloadJobError::Parse)?;
        download_job.validate()?;
        Ok(download_job)
    }

    pub(crate) fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub(crate) fn validate(&self) -> Result<(), DownloadJobError> {
        if !is_valid_huggingface_id(&self.huggingface_id) {
            return Err(DownloadJobError::InvalidHuggingFaceId);
        }
        if !is_valid_immutable_revision(&self.revision) {
            return Err(DownloadJobError::InvalidRevision);
        }
        if self.files.len() > MAXIMUM_DOWNLOAD_JOB_FILE_COUNT {
            return Err(DownloadJobError::TooManyFiles);
        }

        let mut normalized_relative_paths = BTreeSet::new();
        let mut expected_bytes_total = 0_u64;
        let mut completed_bytes_total = 0_u64;
        for (file_index, download_file) in self.files.iter().enumerate() {
            if !is_safe_relative_path(&download_file.relative_path) {
                return Err(DownloadJobError::UnsafeRelativePath { file_index });
            }
            let normalized_relative_path = download_file.relative_path.to_ascii_lowercase();
            if has_path_hierarchy_conflict(&normalized_relative_paths, &normalized_relative_path) {
                return Err(DownloadJobError::FilePathHierarchyConflict);
            }
            if !normalized_relative_paths.insert(normalized_relative_path) {
                return Err(DownloadJobError::DuplicateFilePath);
            }
            if !download_file.expected_digest.is_valid() {
                return Err(DownloadJobError::InvalidDigest { file_index });
            }
            if download_file.bytes_on_disk > download_file.expected_bytes {
                return Err(DownloadJobError::InvalidFileProgress { file_index });
            }
            expected_bytes_total = expected_bytes_total
                .checked_add(download_file.expected_bytes)
                .ok_or(DownloadJobError::InconsistentByteTotals)?;
            completed_bytes_total = completed_bytes_total
                .checked_add(download_file.bytes_on_disk)
                .ok_or(DownloadJobError::InconsistentByteTotals)?;
        }
        if self.bytes_total > MAXIMUM_JAVASCRIPT_SAFE_INTEGER
            || self.bytes_total == 0
            || self.bytes_completed > self.bytes_total
            || self.bytes_completed != completed_bytes_total
            || (!self.files.is_empty() && self.bytes_total != expected_bytes_total)
        {
            return Err(DownloadJobError::InconsistentByteTotals);
        }
        if self.files.is_empty() && self.bytes_completed != 0 {
            return Err(DownloadJobError::InconsistentByteTotals);
        }

        match self.state {
            DownloadJobState::CheckingDisk | DownloadJobState::FetchingManifest => {
                if !self.files.is_empty() {
                    return Err(DownloadJobError::InconsistentByteTotals);
                }
            }
            DownloadJobState::Downloading if self.files.is_empty() => {
                return Err(DownloadJobError::InconsistentByteTotals);
            }
            DownloadJobState::Verifying | DownloadJobState::Publishing => {
                if self.files.is_empty() || self.bytes_completed != self.bytes_total {
                    return Err(DownloadJobError::InconsistentByteTotals);
                }
            }
            DownloadJobState::Downloading | DownloadJobState::Paused | DownloadJobState::Failed => {
            }
        }

        if let Some(current_file_relative_path) = &self.current_file_relative_path
            && (self.state != DownloadJobState::Downloading
                || !self.files.iter().any(|download_file| {
                    download_file.relative_path == *current_file_relative_path
                }))
        {
            return Err(DownloadJobError::InvalidCurrentFile);
        }
        if (self.state == DownloadJobState::Failed) != self.error_code.is_some() {
            return Err(DownloadJobError::InvalidErrorState);
        }
        Ok(())
    }

    pub(crate) fn reconcile_file_progress(
        &mut self,
        bytes_on_disk_by_file: &[u64],
        updated_at_unix_millis: u64,
        should_pause_interrupted_state: bool,
    ) -> Result<(), DownloadJobError> {
        if bytes_on_disk_by_file.len() != self.files.len() {
            return Err(DownloadJobError::InconsistentByteTotals);
        }
        let mut bytes_completed = 0_u64;
        for (file_index, (download_file, bytes_on_disk)) in self
            .files
            .iter_mut()
            .zip(bytes_on_disk_by_file.iter().copied())
            .enumerate()
        {
            if bytes_on_disk > download_file.expected_bytes {
                return Err(DownloadJobError::InvalidFileProgress { file_index });
            }
            download_file.bytes_on_disk = bytes_on_disk;
            bytes_completed = bytes_completed
                .checked_add(bytes_on_disk)
                .ok_or(DownloadJobError::InconsistentByteTotals)?;
        }
        self.bytes_completed = bytes_completed;
        self.current_file_relative_path = None;
        if should_pause_interrupted_state && self.state.is_interrupted_by_restart() {
            self.state = DownloadJobState::Paused;
            self.error_code = None;
        }
        self.updated_at_unix_millis = updated_at_unix_millis;
        self.validate()
    }

    pub(crate) fn force_paused(&mut self, updated_at_unix_millis: u64) {
        self.state = DownloadJobState::Paused;
        self.error_code = None;
        self.current_file_relative_path = None;
        self.updated_at_unix_millis = updated_at_unix_millis;
    }

    #[must_use]
    pub fn huggingface_id(&self) -> &str {
        &self.huggingface_id
    }

    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    #[must_use]
    pub const fn state(&self) -> DownloadJobState {
        self.state
    }

    #[must_use]
    pub const fn bytes_completed(&self) -> u64 {
        self.bytes_completed
    }

    #[must_use]
    pub const fn bytes_total(&self) -> u64 {
        self.bytes_total
    }

    #[must_use]
    pub const fn remaining_bytes(&self) -> u64 {
        self.bytes_total - self.bytes_completed
    }

    #[must_use]
    pub fn has_exact_manifest(&self) -> bool {
        !self.files.is_empty()
    }

    pub(crate) fn staging_directory(&self, models_directory: &Path) -> PathBuf {
        let mut identity_components = self.huggingface_id.split('/');
        let organization = identity_components.next().unwrap_or_default();
        let model_name = identity_components.next().unwrap_or_default();
        models_directory
            .join(".incomplete")
            .join(organization)
            .join(model_name)
    }

    #[must_use]
    pub fn current_file_relative_path(&self) -> Option<&str> {
        self.current_file_relative_path.as_deref()
    }

    #[must_use]
    pub fn files(&self) -> &[DownloadJobFile] {
        &self.files
    }

    #[must_use]
    pub const fn error_code(&self) -> Option<DownloadJobPublicErrorCode> {
        self.error_code
    }

    #[must_use]
    pub const fn updated_at_unix_millis(&self) -> u64 {
        self.updated_at_unix_millis
    }
}

impl DownloadJobFile {
    #[must_use]
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    #[must_use]
    pub const fn expected_bytes(&self) -> u64 {
        self.expected_bytes
    }

    #[must_use]
    pub fn expected_digest(&self) -> &DownloadFileDigest {
        &self.expected_digest
    }

    #[must_use]
    pub const fn bytes_on_disk(&self) -> u64 {
        self.bytes_on_disk
    }
}

impl DownloadJobState {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CheckingDisk => "checking_disk",
            Self::FetchingManifest => "fetching_manifest",
            Self::Downloading => "downloading",
            Self::Paused => "paused",
            Self::Verifying => "verifying",
            Self::Publishing => "publishing",
            Self::Failed => "failed",
        }
    }

    const fn is_interrupted_by_restart(self) -> bool {
        matches!(
            self,
            Self::CheckingDisk | Self::FetchingManifest | Self::Downloading | Self::Verifying
        )
    }

    pub(crate) const fn can_replace_with(self, replacement_state: Self) -> bool {
        matches!(
            (self, replacement_state),
            (
                Self::CheckingDisk,
                Self::CheckingDisk | Self::FetchingManifest
            ) | (Self::CheckingDisk, Self::Failed)
                | (
                    Self::FetchingManifest,
                    Self::FetchingManifest | Self::Paused | Self::Failed
                )
                | (
                    Self::Downloading,
                    Self::Downloading | Self::Paused | Self::Verifying | Self::Failed
                )
                | (
                    Self::Paused,
                    Self::Paused | Self::FetchingManifest | Self::Downloading | Self::Failed
                )
                | (
                    Self::Verifying,
                    Self::Verifying | Self::Paused | Self::Publishing | Self::Failed
                )
                | (Self::Publishing, Self::Publishing)
                | (
                    Self::Failed,
                    Self::Failed | Self::FetchingManifest | Self::Downloading
                )
        )
    }
}

impl DownloadJobPublicErrorCode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LibraryBusy => "library_busy",
            Self::CatalogEntryNotFound => "catalog_entry_not_found",
            Self::ModelNotPublic => "model_not_public",
            Self::InsufficientDisk => "insufficient_disk",
            Self::DownloadGated => "download_gated",
            Self::ChecksumMismatch => "checksum_mismatch",
            Self::DownloadFailed => "download_failed",
            Self::ModelAlreadyPresent => "model_already_present",
        }
    }
}
