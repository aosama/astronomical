//! The free disk figure the Models page shows comes from the same capacity
//! reading the download list admits models against, and is absent, never zero,
//! when that reading fails.

use std::{io, path::Path, sync::Arc};

use tempfile::TempDir;

use crate::{
    DaemonPerformanceAttributionLog, DiskCapacityQuery, DownloadCatalog,
    DownloadPublicationRefresh, HubHttpRequest, HubPayloadFuture, HubPayloadRequest,
    HubPayloadTransport, HubTransport, HubTransportError, HubTransportFuture,
    LibraryDownloadCoordinator,
};

const FREE_BYTES_ON_THE_TEST_DISK: u64 = 388_694_573_056;

struct ScriptedCapacityQuery {
    free_bytes: Option<u64>,
}

impl DiskCapacityQuery for ScriptedCapacityQuery {
    fn available_space_bytes(&self, _existing_same_volume_path: &Path) -> io::Result<u64> {
        self.free_bytes
            .ok_or_else(|| io::Error::new(io::ErrorKind::PermissionDenied, "volume unreadable"))
    }
}

struct OfflineTransport;

impl HubTransport for OfflineTransport {
    fn execute(&self, _request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async { Err(HubTransportError::new("no Hub in this test")) })
    }
}

impl HubPayloadTransport for OfflineTransport {
    fn execute_payload(&self, _request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async { Err(HubTransportError::new("no Hub in this test")) })
    }
}

struct IgnoreDiscoveryRefresh;

impl DownloadPublicationRefresh for IgnoreDiscoveryRefresh {
    fn refresh(
        &self,
        _published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn refresh_after_removal(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

fn coordinator_reading(
    free_bytes: Option<u64>,
    test_directory: &TempDir,
) -> LibraryDownloadCoordinator {
    LibraryDownloadCoordinator::new(
        Arc::new(DownloadCatalog::empty()),
        test_directory.path().join("models"),
        Arc::new(ScriptedCapacityQuery { free_bytes }),
        Arc::new(OfflineTransport),
        Arc::new(OfflineTransport),
        Arc::new(IgnoreDiscoveryRefresh),
        DaemonPerformanceAttributionLog::open(test_directory.path(), false)
            .expect("disabled attribution should construct"),
    )
}

#[test]
fn should_report_the_free_bytes_the_capacity_query_reads() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let coordinator = coordinator_reading(Some(FREE_BYTES_ON_THE_TEST_DISK), &test_directory);

    assert_eq!(
        coordinator.models_directory_free_bytes(),
        Some(FREE_BYTES_ON_THE_TEST_DISK)
    );
}

#[test]
fn should_report_a_full_disk_as_zero_free_bytes() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let coordinator = coordinator_reading(Some(0), &test_directory);

    assert_eq!(coordinator.models_directory_free_bytes(), Some(0));
}

#[test]
fn should_report_no_figure_when_the_disk_cannot_be_read() {
    let test_directory = TempDir::new().expect("temporary directory should be available");
    let coordinator = coordinator_reading(None, &test_directory);

    assert_eq!(coordinator.models_directory_free_bytes(), None);
}
