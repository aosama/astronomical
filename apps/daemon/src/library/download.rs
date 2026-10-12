//! Download pipeline: coordination, preflights, publication, and progress.

pub(crate) mod download_coordinator;
pub(crate) mod download_disk_preflight;
pub(crate) mod download_endpoint;
pub(crate) mod download_executable_preflight;
pub(crate) mod download_manifest_preflight;
pub(crate) mod download_path_selection;
pub(crate) mod download_progress_snapshot;
pub(crate) mod download_publication;
pub(crate) mod download_publication_provenance;
pub(crate) mod download_publication_reconciliation;

#[cfg(test)]
mod download_disk_preflight_tests;
#[cfg(test)]
mod download_executable_preflight_tests;
#[cfg(test)]
mod download_manifest_preflight_tests;

pub(crate) mod job;
pub(crate) mod transfer;

pub use download_coordinator::{LibraryDownloadCoordinator, LibraryDownloadCoordinatorError};
pub use download_disk_preflight::{
    DiskCapacityQuery, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, Fs4DiskCapacityQuery,
};
pub(crate) use download_endpoint::library_download_routes;
pub use download_executable_preflight::{
    DownloadExecutablePreflight, DownloadExecutablePreflightError,
};
pub use download_manifest_preflight::{DownloadManifestPreflight, DownloadManifestPreflightError};
pub use download_path_selection::DownloadPathSelection;
pub use download_progress_snapshot::DownloadProgressSnapshot;
pub use download_publication::{
    DownloadPublication, DownloadPublicationError, DownloadPublicationRefresh,
};
