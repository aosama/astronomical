//! Attributed transition from verified payloads to restart-safe publication intent.

use crate::library::{DownloadJob, DownloadJobStore, DownloadJobStoreError};
use crate::{
    DaemonPerformanceAttributionLog, DaemonPerformanceMeasurement, DaemonPerformanceOperation,
};

pub(in crate::library::download) async fn reconcile_publication_intent(
    job_store: DownloadJobStore,
    attribution_log: &DaemonPerformanceAttributionLog,
    publication_job: DownloadJob,
) -> Result<Result<(), DownloadJobStoreError>, std::io::Error> {
    let measurement_job = publication_job.clone();
    attribution_log
        .measure_blocking_operation(
            DaemonPerformanceOperation::Verification,
            move || job_store.replace_current_for_publication(&publication_job),
            move |reconciliation_outcome| {
                let measurement = if reconciliation_outcome.is_ok() {
                    DaemonPerformanceMeasurement::success()
                } else {
                    DaemonPerformanceMeasurement::failure()
                };
                measurement
                    .with_verification(
                        measurement_job.huggingface_id(),
                        measurement_job.revision(),
                        measurement_job.files().len(),
                        measurement_job.bytes_total(),
                    )
                    .unwrap_or_else(|_| DaemonPerformanceMeasurement::failure())
            },
        )
        .await
}
