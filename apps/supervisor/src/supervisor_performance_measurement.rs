//! Builds and validates operation-specific supervisor performance measurements.

use std::io;

use crate::supervisor_download_attribution::{
    SupervisorDownloadMeasurementDetail, SupervisorDownloadOperationDetail, is_safe_relative_path,
};
use crate::{SupervisorPerformanceMeasurement, SupervisorPerformanceOperation};

impl SupervisorPerformanceMeasurement {
    pub fn with_disk_preflight(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        required_bytes: u64,
        available_bytes: u64,
    ) -> io::Result<Self> {
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::DiskPreflight {
                required_bytes,
                available_bytes,
            },
        )?);
        Ok(self)
    }

    pub fn with_manifest_fetch(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        manifest_file_count: usize,
        manifest_total_bytes: u64,
    ) -> io::Result<Self> {
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::ManifestFetch {
                manifest_file_count,
                manifest_total_bytes,
            },
        )?);
        Ok(self)
    }

    pub fn with_file_transfer(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        relative_file_path: impl Into<String>,
        resume_offset_bytes: u64,
        transferred_bytes: u64,
    ) -> io::Result<Self> {
        let relative_file_path = relative_file_path.into();
        if !is_safe_relative_path(&relative_file_path) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "supervisor attribution requires a bounded safe relative file path",
            ));
        }
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::FileTransfer {
                relative_file_path,
                resume_offset_bytes,
                transferred_bytes,
            },
        )?);
        Ok(self)
    }

    pub fn with_verification(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
        verified_file_count: usize,
        verified_bytes: u64,
    ) -> io::Result<Self> {
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::Verification {
                verified_file_count,
                verified_bytes,
            },
        )?);
        Ok(self)
    }

    pub fn with_publication(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
    ) -> io::Result<Self> {
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::Publication {},
        )?);
        Ok(self)
    }

    pub fn with_discovery_refresh(
        mut self,
        huggingface_id: impl Into<String>,
        revision: impl Into<String>,
    ) -> io::Result<Self> {
        self.download_detail = Some(SupervisorDownloadMeasurementDetail::new(
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::DiscoveryRefresh {},
        )?);
        Ok(self)
    }

    pub(super) fn matches_operation(&self, operation: SupervisorPerformanceOperation) -> bool {
        matches!(
            (
                operation,
                self.download_detail
                    .as_ref()
                    .map(|detail| &detail.operation_detail)
            ),
            (SupervisorPerformanceOperation::LibraryCatalogLoad, None)
                | (
                    SupervisorPerformanceOperation::DiskPreflight,
                    Some(SupervisorDownloadOperationDetail::DiskPreflight { .. })
                )
                | (
                    SupervisorPerformanceOperation::ManifestFetch,
                    Some(SupervisorDownloadOperationDetail::ManifestFetch { .. })
                )
                | (
                    SupervisorPerformanceOperation::FileTransfer,
                    Some(SupervisorDownloadOperationDetail::FileTransfer { .. })
                )
                | (
                    SupervisorPerformanceOperation::Verification,
                    Some(SupervisorDownloadOperationDetail::Verification { .. })
                )
                | (
                    SupervisorPerformanceOperation::Publication,
                    Some(SupervisorDownloadOperationDetail::Publication { .. })
                )
                | (
                    SupervisorPerformanceOperation::DiscoveryRefresh,
                    Some(SupervisorDownloadOperationDetail::DiscoveryRefresh { .. })
                )
                | (
                    SupervisorPerformanceOperation::QwenThinkingChannelSeedLoad,
                    None
                )
        )
    }
}
