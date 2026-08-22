//! Disk-capacity admission for staged Library downloads.

use std::{io, path::Path, path::PathBuf};

use thiserror::Error;

use super::DownloadJob;

const ONE_PERCENT_DIVISOR: u64 = 100;

/// Queries free bytes on the volume containing an existing path.
pub trait DiskCapacityQuery: Send + Sync {
    fn available_space_bytes(&self, existing_same_volume_path: &Path) -> io::Result<u64>;
}

/// Production capacity query backed by the operating system filesystem view.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fs4DiskCapacityQuery;

impl DiskCapacityQuery for Fs4DiskCapacityQuery {
    fn available_space_bytes(&self, existing_same_volume_path: &Path) -> io::Result<u64> {
        // Querying an existing ancestor keeps the answer on the destination volume even before
        // the hidden staging tree has been created.
        fs4::available_space(existing_same_volume_path)
    }
}

/// Applies download-specific capacity policy through an injectable volume query.
#[derive(Clone, Debug)]
pub struct DownloadDiskPreflight<CapacityQuery = Fs4DiskCapacityQuery> {
    capacity_query: CapacityQuery,
}

impl DownloadDiskPreflight<Fs4DiskCapacityQuery> {
    #[must_use]
    pub const fn production() -> Self {
        Self {
            capacity_query: Fs4DiskCapacityQuery,
        }
    }
}

impl<CapacityQuery> DownloadDiskPreflight<CapacityQuery>
where
    CapacityQuery: DiskCapacityQuery,
{
    #[must_use]
    pub const fn new(capacity_query: CapacityQuery) -> Self {
        Self { capacity_query }
    }

    #[must_use]
    pub const fn capacity_query(&self) -> &CapacityQuery {
        &self.capacity_query
    }

    /// Checks the release estimate with a minimum-one-byte one-percent staging margin.
    pub fn check_initial_download(
        &self,
        existing_same_volume_path: &Path,
        catalog_approximate_bytes: u64,
    ) -> Result<DownloadDiskCapacityCheck, DownloadDiskPreflightError> {
        let margin_bytes = (catalog_approximate_bytes / ONE_PERCENT_DIVISOR)
            .saturating_add(u64::from(
                catalog_approximate_bytes % ONE_PERCENT_DIVISOR != 0,
            ))
            .max(1);
        let required_bytes = catalog_approximate_bytes.checked_add(margin_bytes).ok_or(
            DownloadDiskPreflightError::RequiredBytesOverflow {
                catalog_approximate_bytes,
                margin_bytes,
            },
        )?;

        self.check_required_bytes(existing_same_volume_path, required_bytes)
    }

    /// Checks only bytes that remain after the exact manifest and staged files are known.
    pub fn check_job_remaining_bytes(
        &self,
        existing_same_volume_path: &Path,
        download_job: &DownloadJob,
    ) -> Result<DownloadDiskCapacityCheck, DownloadDiskPreflightError> {
        if !download_job.has_exact_manifest() {
            return Err(DownloadDiskPreflightError::ExactManifestRequired);
        }
        self.check_required_bytes(existing_same_volume_path, download_job.remaining_bytes())
    }

    fn check_required_bytes(
        &self,
        existing_same_volume_path: &Path,
        required_bytes: u64,
    ) -> Result<DownloadDiskCapacityCheck, DownloadDiskPreflightError> {
        let available_bytes = self
            .capacity_query
            .available_space_bytes(existing_same_volume_path)
            .map_err(|source| DownloadDiskPreflightError::QueryCapacity {
                path: existing_same_volume_path.to_path_buf(),
                required_bytes,
                source,
            })?;
        if required_bytes > available_bytes {
            return Err(DownloadDiskPreflightError::InsufficientSpace {
                required_bytes,
                available_bytes,
            });
        }

        Ok(DownloadDiskCapacityCheck {
            required_bytes,
            available_bytes,
        })
    }
}

/// Capacity evidence retained for attribution and later orchestration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DownloadDiskCapacityCheck {
    required_bytes: u64,
    available_bytes: u64,
}

impl DownloadDiskCapacityCheck {
    #[must_use]
    pub const fn required_bytes(&self) -> u64 {
        self.required_bytes
    }

    #[must_use]
    pub const fn available_bytes(&self) -> u64 {
        self.available_bytes
    }
}

/// Typed disk-query, arithmetic, or admission failure.
#[derive(Debug, Error)]
pub enum DownloadDiskPreflightError {
    #[error("exact manifest metadata is required for the remaining-byte disk check")]
    ExactManifestRequired,
    #[error(
        "download disk requirement overflowed: catalog estimate {catalog_approximate_bytes} bytes, margin {margin_bytes} bytes"
    )]
    RequiredBytesOverflow {
        catalog_approximate_bytes: u64,
        margin_bytes: u64,
    },
    #[error(
        "failed to query download capacity at {path:?} for {required_bytes} required bytes: {source}"
    )]
    QueryCapacity {
        path: PathBuf,
        required_bytes: u64,
        #[source]
        source: io::Error,
    },
    #[error(
        "insufficient download space: required {required_bytes} bytes, available {available_bytes} bytes"
    )]
    InsufficientSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
}
