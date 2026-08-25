//! Switchable start/end attribution for supervisor-owned operations.

use std::{
    fs::OpenOptions,
    future::Future,
    io::{self, BufWriter, Write},
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::supervisor_download_attribution::{
    SupervisorDownloadMeasurementDetail, SupervisorDownloadOperationDetail,
};
use crate::supervisor_performance_record::{
    SupervisorPerformanceAttributionRecord, SupervisorPerformanceOutcome, current_unix_epoch_millis,
};

const SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME: &str =
    "supervisor-performance-attribution.jsonl";

type SupervisorPerformanceClock = dyn Fn() -> io::Result<u64> + Send + Sync;

/// Supervisor-owned operation names recorded by the common attribution log.
#[derive(Clone, Copy, Debug)]
pub enum SupervisorPerformanceOperation {
    LibraryCatalogLoad,
    DiskPreflight,
    ManifestFetch,
    FileTransfer,
    Verification,
    Publication,
    DiscoveryRefresh,
    QwenThinkingChannelSeedLoad,
}

impl SupervisorPerformanceOperation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::LibraryCatalogLoad => "library_catalog_load",
            Self::DiskPreflight => "disk_preflight",
            Self::ManifestFetch => "manifest_fetch",
            Self::FileTransfer => "file_transfer",
            Self::Verification => "verification",
            Self::Publication => "publication",
            Self::DiscoveryRefresh => "discovery_refresh",
            Self::QwenThinkingChannelSeedLoad => "qwen_thinking_channel_seed_load",
        }
    }
}

/// Explicit outcome and operation-specific metadata for one measured boundary.
#[derive(Clone, Debug)]
pub struct SupervisorPerformanceMeasurement {
    pub(crate) outcome: SupervisorPerformanceOutcome,
    pub(crate) catalog_entry_count: Option<usize>,
    pub(crate) download_detail: Option<SupervisorDownloadMeasurementDetail>,
}

impl SupervisorPerformanceMeasurement {
    pub(crate) fn validated_disk_preflight(
        is_success: bool,
        huggingface_id: &str,
        revision: &str,
        required_bytes: u64,
        available_bytes: u64,
    ) -> Self {
        Self::validated_download_measurement(
            is_success,
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::DiskPreflight {
                required_bytes,
                available_bytes,
            },
        )
    }

    pub(crate) fn validated_manifest_fetch(
        is_success: bool,
        huggingface_id: &str,
        revision: &str,
        manifest_file_count: usize,
        manifest_total_bytes: u64,
    ) -> Self {
        Self::validated_download_measurement(
            is_success,
            huggingface_id,
            revision,
            SupervisorDownloadOperationDetail::ManifestFetch {
                manifest_file_count,
                manifest_total_bytes,
            },
        )
    }

    fn validated_download_measurement(
        is_success: bool,
        huggingface_id: &str,
        revision: &str,
        operation_detail: SupervisorDownloadOperationDetail,
    ) -> Self {
        Self {
            outcome: if is_success {
                SupervisorPerformanceOutcome::Success
            } else {
                SupervisorPerformanceOutcome::Failure
            },
            catalog_entry_count: None,
            download_detail: Some(SupervisorDownloadMeasurementDetail::validated(
                huggingface_id,
                revision,
                operation_detail,
            )),
        }
    }

    #[must_use]
    pub const fn success() -> Self {
        Self::with_outcome(SupervisorPerformanceOutcome::Success)
    }

    #[must_use]
    pub const fn failure() -> Self {
        Self::with_outcome(SupervisorPerformanceOutcome::Failure)
    }

    #[must_use]
    pub const fn paused() -> Self {
        Self::with_outcome(SupervisorPerformanceOutcome::Paused)
    }

    #[must_use]
    pub const fn cancelled() -> Self {
        Self::with_outcome(SupervisorPerformanceOutcome::Cancelled)
    }

    #[must_use]
    pub const fn successful_catalog_load(catalog_entry_count: usize) -> Self {
        Self {
            outcome: SupervisorPerformanceOutcome::Success,
            catalog_entry_count: Some(catalog_entry_count),
            download_detail: None,
        }
    }

    const fn with_outcome(outcome: SupervisorPerformanceOutcome) -> Self {
        Self {
            outcome,
            catalog_entry_count: None,
            download_detail: None,
        }
    }
}

struct SupervisorPerformanceAttributionSink {
    writer: Mutex<Box<dyn Write + Send>>,
    unix_epoch_millis: Arc<SupervisorPerformanceClock>,
}

/// Cloneable append-only log whose disabled state owns neither a writer nor a clock.
#[derive(Clone)]
pub struct SupervisorPerformanceAttributionLog {
    enabled_sink: Option<Arc<SupervisorPerformanceAttributionSink>>,
}

impl SupervisorPerformanceAttributionLog {
    /// Creates a no-overhead log for tests and application variants without diagnostics.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { enabled_sink: None }
    }

    pub fn open(log_directory: &Path, performance_attribution_enabled: bool) -> io::Result<Self> {
        if !performance_attribution_enabled {
            return Ok(Self { enabled_sink: None });
        }
        let attribution_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_directory.join(SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME))?;
        Ok(Self::enabled(
            BufWriter::new(attribution_file),
            Arc::new(current_unix_epoch_millis),
        ))
    }

    /// Keeps writer failures deterministic without depending on filesystem behavior.
    pub fn from_writer(writer: impl Write + Send + 'static) -> Self {
        Self::enabled(BufWriter::new(writer), Arc::new(current_unix_epoch_millis))
    }

    /// Keeps clock failures deterministic without changing production time ownership.
    pub fn from_writer_and_clock(
        writer: impl Write + Send + 'static,
        unix_epoch_millis: impl Fn() -> io::Result<u64> + Send + Sync + 'static,
    ) -> Self {
        Self::enabled(BufWriter::new(writer), Arc::new(unix_epoch_millis))
    }

    /// Proves the disabled branch discards test collaborators before any operation executes.
    pub fn from_writer_and_clock_when_enabled(
        writer: impl Write + Send + 'static,
        unix_epoch_millis: impl Fn() -> io::Result<u64> + Send + Sync + 'static,
        performance_attribution_enabled: bool,
    ) -> Self {
        if !performance_attribution_enabled {
            return Self { enabled_sink: None };
        }
        Self::enabled(BufWriter::new(writer), Arc::new(unix_epoch_millis))
    }

    /// Measures an operation result and appends one flushed JSON record when enabled.
    pub fn measure_operation<OperationOutput>(
        &self,
        operation: SupervisorPerformanceOperation,
        measured_operation: impl FnOnce() -> OperationOutput,
        describe_measurement: impl FnOnce(&OperationOutput) -> SupervisorPerformanceMeasurement,
    ) -> io::Result<OperationOutput> {
        let Some(enabled_sink) = &self.enabled_sink else {
            return Ok(measured_operation());
        };

        let started_at_unix_millis = (enabled_sink.unix_epoch_millis)()?;
        let started_at = Instant::now();
        let operation_output = measured_operation();
        Self::finish_measurement(
            enabled_sink,
            operation,
            started_at_unix_millis,
            started_at,
            describe_measurement(&operation_output),
        )?;
        Ok(operation_output)
    }

    /// The operation future runs without owning the synchronous writer lock.
    pub async fn measure_async_operation<OperationOutput, OperationFuture>(
        &self,
        operation: SupervisorPerformanceOperation,
        measured_operation: impl FnOnce() -> OperationFuture,
        describe_measurement: impl FnOnce(&OperationOutput) -> SupervisorPerformanceMeasurement,
    ) -> io::Result<OperationOutput>
    where
        OperationFuture: Future<Output = OperationOutput>,
    {
        let Some(enabled_sink) = &self.enabled_sink else {
            return Ok(measured_operation().await);
        };

        let started_at_unix_millis = (enabled_sink.unix_epoch_millis)()?;
        let started_at = Instant::now();
        let operation_output = measured_operation().await;
        let measurement = describe_measurement(&operation_output);
        self.record_async_measurement(
            Arc::clone(enabled_sink),
            operation,
            started_at_unix_millis,
            started_at,
            measurement,
        )
        .await?;
        Ok(operation_output)
    }

    /// Measures request-path work without allowing diagnostic failures to alter its output.
    pub async fn measure_async_operation_best_effort<OperationOutput, OperationFuture>(
        &self,
        operation: SupervisorPerformanceOperation,
        measured_operation: impl FnOnce() -> OperationFuture,
        describe_measurement: impl FnOnce(&OperationOutput) -> SupervisorPerformanceMeasurement,
    ) -> OperationOutput
    where
        OperationFuture: Future<Output = OperationOutput>,
    {
        let Some(enabled_sink) = &self.enabled_sink else {
            return measured_operation().await;
        };
        let started_at_unix_millis = match (enabled_sink.unix_epoch_millis)() {
            Ok(started_at_unix_millis) => started_at_unix_millis,
            Err(attribution_error) => {
                tracing::warn!(
                    operation = operation.as_str(),
                    error = %attribution_error,
                    "supervisor performance attribution could not start"
                );
                return measured_operation().await;
            }
        };
        let started_at = Instant::now();
        let operation_output = measured_operation().await;
        let measurement = describe_measurement(&operation_output);
        if let Err(attribution_error) = self
            .record_async_measurement(
                Arc::clone(enabled_sink),
                operation,
                started_at_unix_millis,
                started_at,
                measurement,
            )
            .await
        {
            // Optional diagnostics cannot become a hidden model-steering dependency.
            tracing::warn!(
                operation = operation.as_str(),
                error = %attribution_error,
                "supervisor performance attribution could not finish"
            );
        }
        operation_output
    }

    /// Runs synchronous disk or filesystem work away from the asynchronous executor.
    pub async fn measure_blocking_operation<OperationOutput>(
        &self,
        operation: SupervisorPerformanceOperation,
        measured_operation: impl FnOnce() -> OperationOutput + Send + 'static,
        describe_measurement: impl FnOnce(&OperationOutput) -> SupervisorPerformanceMeasurement,
    ) -> io::Result<OperationOutput>
    where
        OperationOutput: Send + 'static,
    {
        let enabled_timing = match self.enabled_sink.as_ref() {
            Some(enabled_sink) => Some((
                Arc::clone(enabled_sink),
                (enabled_sink.unix_epoch_millis)()?,
                Instant::now(),
            )),
            None => None,
        };
        let operation_output = tokio::task::spawn_blocking(measured_operation)
            .await
            .map_err(io::Error::other)?;
        if let Some((enabled_sink, started_at_unix_millis, started_at)) = enabled_timing {
            self.record_async_measurement(
                enabled_sink,
                operation,
                started_at_unix_millis,
                started_at,
                describe_measurement(&operation_output),
            )
            .await?;
        }
        Ok(operation_output)
    }

    async fn record_async_measurement(
        &self,
        enabled_sink: Arc<SupervisorPerformanceAttributionSink>,
        operation: SupervisorPerformanceOperation,
        started_at_unix_millis: u64,
        started_at: Instant,
        measurement: SupervisorPerformanceMeasurement,
    ) -> io::Result<()> {
        let elapsed_nanoseconds =
            u64::try_from(started_at.elapsed().as_nanos()).map_err(io::Error::other)?;
        let ended_at_unix_millis = (enabled_sink.unix_epoch_millis)()?;
        tokio::task::spawn_blocking(move || {
            Self::record_completed_measurement(
                &enabled_sink,
                operation,
                started_at_unix_millis,
                ended_at_unix_millis,
                elapsed_nanoseconds,
                measurement,
            )
        })
        .await
        .map_err(io::Error::other)??;
        Ok(())
    }

    fn enabled(
        writer: impl Write + Send + 'static,
        unix_epoch_millis: Arc<SupervisorPerformanceClock>,
    ) -> Self {
        Self {
            enabled_sink: Some(Arc::new(SupervisorPerformanceAttributionSink {
                writer: Mutex::new(Box::new(writer)),
                unix_epoch_millis,
            })),
        }
    }

    fn finish_measurement(
        enabled_sink: &SupervisorPerformanceAttributionSink,
        operation: SupervisorPerformanceOperation,
        started_at_unix_millis: u64,
        started_at: Instant,
        measurement: SupervisorPerformanceMeasurement,
    ) -> io::Result<()> {
        let elapsed_nanoseconds =
            u64::try_from(started_at.elapsed().as_nanos()).map_err(io::Error::other)?;
        let ended_at_unix_millis = (enabled_sink.unix_epoch_millis)()?;
        Self::record_completed_measurement(
            enabled_sink,
            operation,
            started_at_unix_millis,
            ended_at_unix_millis,
            elapsed_nanoseconds,
            measurement,
        )
    }

    fn record_completed_measurement(
        enabled_sink: &SupervisorPerformanceAttributionSink,
        operation: SupervisorPerformanceOperation,
        started_at_unix_millis: u64,
        ended_at_unix_millis: u64,
        elapsed_nanoseconds: u64,
        measurement: SupervisorPerformanceMeasurement,
    ) -> io::Result<()> {
        if !measurement.matches_operation(operation) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "supervisor attribution operation and measurement detail do not match",
            ));
        }
        let attribution_record = SupervisorPerformanceAttributionRecord::new(
            operation.as_str(),
            started_at_unix_millis,
            ended_at_unix_millis,
            elapsed_nanoseconds,
            measurement.outcome,
            measurement.catalog_entry_count,
            measurement.download_detail.as_ref(),
        );
        Self::record(enabled_sink, &attribution_record)
    }

    fn record(
        enabled_sink: &SupervisorPerformanceAttributionSink,
        attribution_record: &SupervisorPerformanceAttributionRecord<'_>,
    ) -> io::Result<()> {
        // Serialization stays outside the critical section so concurrent operations serialize
        // only the append required to keep each JSON line intact.
        let serialized_record = serde_json::to_vec(attribution_record).map_err(io::Error::other)?;
        let mut writer = enabled_sink
            .writer
            .lock()
            .map_err(|_| io::Error::other("supervisor attribution writer lock was poisoned"))?;
        writer.write_all(&serialized_record)?;
        writer.write_all(b"\n")?;
        writer.flush()
    }
}
