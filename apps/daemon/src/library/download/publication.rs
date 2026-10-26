//! Atomic staging publication with restart-safe discovery refresh completion.

pub(crate) mod provenance;
pub(crate) mod reconciliation;

use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use thiserror::Error;

use crate::library::{DownloadJobStore, DownloadJobStoreError};
use crate::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};

pub trait DownloadPublicationRefresh: Send + Sync {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub struct DownloadPublication {
    job_store: DownloadJobStore,
    attribution_log: DaemonPerformanceAttributionLog,
}

#[derive(Debug, Error)]
pub enum DownloadPublicationError {
    #[error("durable publication failed: {0}")]
    JobStore(#[from] DownloadJobStoreError),
    #[error("daemon performance attribution failed: {0}")]
    Attribution(#[source] std::io::Error),
    #[error("publication task failed: {0}")]
    Task(#[source] tokio::task::JoinError),
}

impl DownloadPublication {
    #[must_use]
    pub fn new(
        job_store: DownloadJobStore,
        attribution_log: DaemonPerformanceAttributionLog,
    ) -> Self {
        Self {
            job_store,
            attribution_log,
        }
    }

    pub async fn publish(
        &self,
        discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    ) -> Result<PathBuf, DownloadPublicationError> {
        let download_job = self
            .job_store
            .load()?
            .ok_or(DownloadJobStoreError::JobNotFound)?;
        let huggingface_id = download_job.huggingface_id().to_owned();
        let revision = download_job.revision().to_owned();
        let job_store = self.job_store.clone();
        let measured_refresh = MeasuredDiscoveryRefresh {
            discovery_refresh,
            attribution_log: self.attribution_log.clone(),
            huggingface_id: huggingface_id.clone(),
            revision: revision.clone(),
        };
        // Two failure layers: the outer wait fails when the attribution
        // machinery itself errors; the inner Result is the publication outcome.
        let publication_join = self
            .attribution_log
            .measure_blocking_operation(
                DaemonPerformanceOperation::Publication,
                move || job_store.publish_current_job(&measured_refresh),
                |publication_outcome| {
                    let measurement = if publication_outcome.is_ok() {
                        DaemonPerformanceMeasurement::success()
                    } else {
                        DaemonPerformanceMeasurement::failure()
                    };
                    measurement
                        .with_publication(&huggingface_id, &revision)
                        .unwrap_or_else(|_| DaemonPerformanceMeasurement::failure())
                },
            )
            .await;
        let publication_result = match publication_join {
            Ok(publication_result) => publication_result,
            Err(attribution_error) => {
                return Err(DownloadPublicationError::Attribution(attribution_error));
            }
        };
        let publication_outcome = publication_result?;
        Ok(publication_outcome)
    }
}

struct MeasuredDiscoveryRefresh {
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    attribution_log: DaemonPerformanceAttributionLog,
    huggingface_id: String,
    revision: String,
}

impl DownloadPublicationRefresh for MeasuredDiscoveryRefresh {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        let refresh_io_outcome = self.attribution_log.measure_operation(
            DaemonPerformanceOperation::DiscoveryRefresh,
            || self.discovery_refresh.refresh(published_directory),
            |refresh_outcome| {
                let measurement = if refresh_outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                };
                measurement
                    .with_discovery_refresh(&self.huggingface_id, &self.revision)
                    .unwrap_or_else(|_| DaemonPerformanceMeasurement::failure())
            },
        );
        match refresh_io_outcome {
            // The attribution layer wraps the refresh result in its own
            // io::Result to report sink failures; the refresh result itself is
            // the caller-facing outcome.
            Ok(refresh_outcome) => refresh_outcome,
            Err(source) => Err(Box::new(source) as Box<dyn Error + Send + Sync>),
        }
    }
}
