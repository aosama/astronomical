//! Download pipeline: coordination, preflights, publication, and progress.

pub(crate) mod coordinator;
pub(crate) mod disk_preflight;
pub(crate) mod endpoint;
pub(crate) mod executable_preflight;
pub(crate) mod manifest_preflight;
pub(crate) mod path_selection;
pub(crate) mod progress_snapshot;
pub(crate) mod publication;

#[cfg(test)]
mod disk_preflight_tests;
#[cfg(test)]
mod executable_preflight_tests;
#[cfg(test)]
mod manifest_preflight_tests;

pub(crate) mod job;
pub(crate) mod transfer;

pub use coordinator::{LibraryDownloadCoordinator, LibraryDownloadCoordinatorError};
pub use disk_preflight::{
    DiskCapacityQuery, DownloadDiskCapacityCheck, DownloadDiskPreflight,
    DownloadDiskPreflightError, Fs4DiskCapacityQuery,
};
pub(crate) use endpoint::library_download_routes;
pub use executable_preflight::{DownloadExecutablePreflight, DownloadExecutablePreflightError};
pub use manifest_preflight::{DownloadManifestPreflight, DownloadManifestPreflightError};
pub use path_selection::DownloadPathSelection;
pub use progress_snapshot::DownloadProgressSnapshot;
pub use publication::{DownloadPublication, DownloadPublicationError, DownloadPublicationRefresh};
