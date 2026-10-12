//! Serialized record shape and wall-clock source for daemon performance attribution.

use std::{
    io,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

use crate::performance::daemon_download_attribution::DaemonDownloadMeasurementDetail;

#[derive(Clone, Copy, Debug)]
pub(crate) enum DaemonPerformanceOutcome {
    Success,
    Failure,
    Paused,
    Cancelled,
}

impl DaemonPerformanceOutcome {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Paused => "paused",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DaemonPerformanceAttributionRecord<'a> {
    operation: &'static str,
    started_at_unix_millis: u64,
    ended_at_unix_millis: u64,
    elapsed_nanoseconds: u64,
    outcome: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    catalog_entry_count: Option<usize>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    download_detail: Option<&'a DaemonDownloadMeasurementDetail>,
}

impl<'a> DaemonPerformanceAttributionRecord<'a> {
    pub(crate) fn new(
        operation: &'static str,
        started_at_unix_millis: u64,
        ended_at_unix_millis: u64,
        elapsed_nanoseconds: u64,
        outcome: DaemonPerformanceOutcome,
        catalog_entry_count: Option<usize>,
        download_detail: Option<&'a DaemonDownloadMeasurementDetail>,
    ) -> Self {
        Self {
            operation,
            started_at_unix_millis,
            ended_at_unix_millis,
            elapsed_nanoseconds,
            outcome: outcome.as_str(),
            catalog_entry_count,
            download_detail,
        }
    }
}

pub(crate) fn current_unix_epoch_millis() -> io::Result<u64> {
    let duration_since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?;
    u64::try_from(duration_since_epoch.as_millis()).map_err(io::Error::other)
}
