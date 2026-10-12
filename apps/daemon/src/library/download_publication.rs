//! Atomic staging publication with restart-safe discovery refresh completion.

use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};

use thiserror::Error;

use super::{DownloadJobStore, DownloadJobStoreError};
use crate::{
    SupervisorPerformanceAttributionLog, SupervisorPerformanceMeasurement,
    SupervisorPerformanceOperation,
};

pub trait DownloadPublicationRefresh: Send + Sync {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>>;
}

pub struct DownloadPublication {
    job_store: DownloadJobStore,
    attribution_log: SupervisorPerformanceAttributionLog,
}

#[derive(Debug, Error)]
pub enum DownloadPublicationError {
    #[error("durable publication failed: {0}")]
    JobStore(#[from] DownloadJobStoreError),
    #[error("supervisor performance attribution failed: {0}")]
    Attribution(#[source] std::io::Error),
    #[error("publication task failed: {0}")]
    Task(#[source] tokio::task::JoinError),
}

impl DownloadPublication {
    #[must_use]
    pub fn new(
        job_store: DownloadJobStore,
        attribution_log: SupervisorPerformanceAttributionLog,
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
        let publication_outcome = self
            .attribution_log
            .measure_blocking_operation(
                SupervisorPerformanceOperation::Publication,
                move || job_store.publish_current_job(&measured_refresh),
                |publication_outcome| {
                    let measurement = if publication_outcome.is_ok() {
                        SupervisorPerformanceMeasurement::success()
                    } else {
                        SupervisorPerformanceMeasurement::failure()
                    };
                    measurement
                        .with_publication(&huggingface_id, &revision)
                        .unwrap_or_else(|_| SupervisorPerformanceMeasurement::failure())
                },
            )
            .await
            .map_err(DownloadPublicationError::Attribution)??;
        Ok(publication_outcome)
    }
}

struct MeasuredDiscoveryRefresh {
    discovery_refresh: Arc<dyn DownloadPublicationRefresh>,
    attribution_log: SupervisorPerformanceAttributionLog,
    huggingface_id: String,
    revision: String,
}

impl DownloadPublicationRefresh for MeasuredDiscoveryRefresh {
    fn refresh(&self, published_directory: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        let refresh_outcome = self.attribution_log.measure_operation(
            SupervisorPerformanceOperation::DiscoveryRefresh,
            || self.discovery_refresh.refresh(published_directory),
            |refresh_outcome| {
                let measurement = if refresh_outcome.is_ok() {
                    SupervisorPerformanceMeasurement::success()
                } else {
                    SupervisorPerformanceMeasurement::failure()
                };
                measurement
                    .with_discovery_refresh(&self.huggingface_id, &self.revision)
                    .unwrap_or_else(|_| SupervisorPerformanceMeasurement::failure())
            },
        )?;
        refresh_outcome
    }
}
