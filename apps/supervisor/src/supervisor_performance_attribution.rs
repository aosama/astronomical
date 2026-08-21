//! Switchable start/end attribution for supervisor-owned operations.

use std::{
    fs::OpenOptions,
    io::{self, BufWriter, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

const SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME: &str =
    "supervisor-performance-attribution.jsonl";

/// Supervisor operation names shared by current and future Library stages.
#[derive(Clone, Copy, Debug)]
pub enum SupervisorPerformanceOperation {
    LibraryCatalogLoad,
}

impl SupervisorPerformanceOperation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::LibraryCatalogLoad => "library_catalog_load",
        }
    }
}

/// Explicit outcome and operation-specific metadata for one measured boundary.
#[derive(Clone, Copy, Debug)]
pub struct SupervisorPerformanceMeasurement {
    outcome: SupervisorPerformanceOutcome,
    catalog_entry_count: Option<usize>,
}

impl SupervisorPerformanceMeasurement {
    #[must_use]
    pub const fn success() -> Self {
        Self {
            outcome: SupervisorPerformanceOutcome::Success,
            catalog_entry_count: None,
        }
    }

    #[must_use]
    pub const fn failure() -> Self {
        Self {
            outcome: SupervisorPerformanceOutcome::Failure,
            catalog_entry_count: None,
        }
    }

    #[must_use]
    pub const fn successful_catalog_load(catalog_entry_count: usize) -> Self {
        Self {
            outcome: SupervisorPerformanceOutcome::Success,
            catalog_entry_count: Some(catalog_entry_count),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum SupervisorPerformanceOutcome {
    Success,
    Failure,
}

impl SupervisorPerformanceOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
        }
    }
}

/// Append-only writer whose disabled state performs no file or clock work.
pub struct SupervisorPerformanceAttributionLog {
    writer: Option<Box<dyn Write + Send>>,
    unix_epoch_millis: Box<dyn Fn() -> io::Result<u64> + Send + Sync>,
}

impl SupervisorPerformanceAttributionLog {
    pub fn open(log_directory: &Path, performance_attribution_enabled: bool) -> io::Result<Self> {
        if !performance_attribution_enabled {
            return Ok(Self {
                writer: None,
                unix_epoch_millis: Box::new(current_unix_epoch_millis),
            });
        }
        let attribution_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_directory.join(SUPERVISOR_PERFORMANCE_ATTRIBUTION_FILE_NAME))?;
        Ok(Self {
            writer: Some(Box::new(BufWriter::new(attribution_file))),
            unix_epoch_millis: Box::new(current_unix_epoch_millis),
        })
    }

    /// Allows deterministic write-failure coverage without relying on filesystem behavior.
    pub fn from_writer(writer: impl Write + Send + 'static) -> Self {
        Self {
            writer: Some(Box::new(BufWriter::new(writer))),
            unix_epoch_millis: Box::new(current_unix_epoch_millis),
        }
    }

    /// Allows deterministic clock-failure coverage without changing production time ownership.
    pub fn from_writer_and_clock(
        writer: impl Write + Send + 'static,
        unix_epoch_millis: impl Fn() -> io::Result<u64> + Send + Sync + 'static,
    ) -> Self {
        Self {
            writer: Some(Box::new(BufWriter::new(writer))),
            unix_epoch_millis: Box::new(unix_epoch_millis),
        }
    }

    /// Measures an operation result and appends one flushed JSON record when enabled.
    pub fn measure_operation<OperationOutput>(
        &mut self,
        operation: SupervisorPerformanceOperation,
        measured_operation: impl FnOnce() -> OperationOutput,
        describe_measurement: impl FnOnce(&OperationOutput) -> SupervisorPerformanceMeasurement,
    ) -> io::Result<OperationOutput> {
        if self.writer.is_none() {
            return Ok(measured_operation());
        }

        let started_at_unix_millis = (self.unix_epoch_millis)()?;
        let started_at = Instant::now();
        let operation_output = measured_operation();
        let elapsed_nanoseconds =
            u64::try_from(started_at.elapsed().as_nanos()).map_err(io::Error::other)?;
        let ended_at_unix_millis = (self.unix_epoch_millis)()?;
        let measurement = describe_measurement(&operation_output);
        let attribution_record = SupervisorPerformanceAttributionRecord {
            operation: operation.as_str(),
            started_at_unix_millis,
            ended_at_unix_millis,
            elapsed_nanoseconds,
            outcome: measurement.outcome.as_str(),
            catalog_entry_count: measurement.catalog_entry_count,
        };
        self.record(&attribution_record)?;
        Ok(operation_output)
    }

    fn record(
        &mut self,
        attribution_record: &SupervisorPerformanceAttributionRecord,
    ) -> io::Result<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Ok(());
        };
        let serialized_record = serde_json::to_vec(attribution_record).map_err(io::Error::other)?;
        writer.write_all(&serialized_record)?;
        writer.write_all(b"\n")?;
        writer.flush()
    }
}

#[derive(Debug, Serialize)]
struct SupervisorPerformanceAttributionRecord {
    operation: &'static str,
    started_at_unix_millis: u64,
    ended_at_unix_millis: u64,
    elapsed_nanoseconds: u64,
    outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog_entry_count: Option<usize>,
}

fn current_unix_epoch_millis() -> io::Result<u64> {
    let duration_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    u64::try_from(duration_since_epoch.as_millis()).map_err(io::Error::other)
}
