//! Download job lifecycle and its durable store.

pub(crate) mod download_job;
pub(crate) mod store;

#[cfg(test)]
mod download_job_tests;
#[cfg(test)]
mod store_tests;

pub use download_job::{
    DownloadJob, DownloadJobError, DownloadJobFile, DownloadJobPublicErrorCode, DownloadJobState,
    yields_library_to,
};
pub use store::DownloadJobStore;
pub use store::error::DownloadJobStoreError;
