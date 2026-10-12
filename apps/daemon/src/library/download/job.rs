//! Download job lifecycle and its durable store.

pub(crate) mod download_job;
pub(crate) mod download_job_store;
pub(crate) mod download_job_store_error;
pub(crate) mod download_job_store_filesystem;
pub(crate) mod download_job_store_lock;
pub(crate) mod download_job_store_publication;

#[cfg(test)]
mod download_job_store_tests;
#[cfg(test)]
mod download_job_tests;

pub use download_job::{
    DownloadJob, DownloadJobError, DownloadJobFile, DownloadJobPublicErrorCode, DownloadJobState,
};
pub use download_job_store::DownloadJobStore;
pub use download_job_store_error::DownloadJobStoreError;
