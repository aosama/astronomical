//! Typed failures from durable download-job filesystem transactions.

use std::path::PathBuf;

use thiserror::Error;

use super::DownloadJobError;

/// Durable-job filesystem or validation failure with retained local diagnostic causes.
#[derive(Debug, Error)]
pub enum DownloadJobStoreError {
    #[error("failed to inspect download job path {path:?}: {source}")]
    InspectPath {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("download storage contains an unsafe filesystem object at {path:?}")]
    UnsafeFilesystemObject { path: PathBuf },
    #[error("failed to create download directory {path:?}: {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open download job {path:?}: {source}")]
    OpenJob {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read download job {path:?}: {source}")]
    ReadJob {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("download job metadata exceeds its resource bound")]
    JobDocumentTooLarge,
    #[error("download job metadata is invalid")]
    InvalidJob(#[from] DownloadJobError),
    #[error("a model download job already exists")]
    JobAlreadyExists,
    #[error("no model download job exists to replace")]
    JobNotFound,
    #[error("replacement download job identity differs from the current job")]
    JobIdentityMismatch,
    #[error("replacement download job would roll durable state backward")]
    InvalidJobReplacement,
    #[error("download job operation lock is unavailable")]
    OperationLockUnavailable,
    #[error("failed to acquire download job transaction lock {path:?}: {source}")]
    AcquireTransactionLock {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("publishing-state recovery requires publication reconciliation")]
    PublishingRecoveryRequired,
    #[error("download job is not UTF-8 JSON: {0}")]
    InvalidUtf8(#[source] std::str::Utf8Error),
    #[error("failed to serialize download job: {0}")]
    SerializeJob(#[source] serde_json::Error),
    #[error("failed to write temporary download job {path:?}: {source}")]
    WriteTemporaryJob {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to atomically replace download job {path:?}: {source}")]
    ReplaceJob {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to atomically create download job {path:?}: {source}")]
    CreateJob {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to synchronize download directory {path:?}: {source}")]
    SynchronizeDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("staged file exceeds its manifest size at {path:?}")]
    StagedFileTooLarge { path: PathBuf },
    #[error("staged file progress differs from the durable resume offset at {path:?}")]
    StagedFileProgressMismatch { path: PathBuf },
    #[error("failed to open staged download file {path:?}: {source}")]
    OpenStagedFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to synchronize staged download file {path:?}: {source}")]
    SynchronizeStagedFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to remove download-owned path {path:?}: {source}")]
    RemovePath {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("download publication has inconsistent staging and destination state")]
    InconsistentPublicationState,
    #[error("the model publication destination already exists")]
    PublishedModelAlreadyExists,
    #[error("failed to serialize immutable publication provenance: {0}")]
    SerializePublicationProvenance(#[source] serde_json::Error),
    #[error("failed to write immutable publication provenance at {path:?}: {source}")]
    WritePublicationProvenance {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to atomically publish model at {path:?}: {source}")]
    PublishModel {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("published model discovery refresh failed: {source}")]
    DiscoveryRefresh {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
