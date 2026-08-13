//! Bounded per-layer evidence for expert source demand and retained-layer reuse.

use serde::Serialize;

#[cfg(feature = "direct-mlx")]
use std::sync::Arc;

/// Request phase whose source traffic or retained hit is being attributed.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertSourceRequestPhase {
    Prefill,
    Decode,
    RetentionTransition,
}

impl ExpertSourceRequestPhase {
    pub(super) const ALL: [Self; 3] = [Self::Prefill, Self::Decode, Self::RetentionTransition];

    pub(super) const fn index(self) -> usize {
        match self {
            Self::Prefill => 0,
            Self::Decode => 1,
            Self::RetentionTransition => 2,
        }
    }
}

/// Fixed aggregate for one validated model layer and one request phase.
#[derive(Clone, Debug, Default)]
pub(crate) struct ExpertSourceLayerPhaseMeasurement {
    pub(super) logical_source_payload_bytes: u64,
    pub(super) maximum_source_page_payload_bytes: u64,
    pub(super) source_interval_count: u64,
    pub(super) source_load_count: u64,
    pub(super) resident_hit_count: u64,
    pub(super) avoided_source_payload_bytes: u64,
    pub(super) avoided_source_interval_count: u64,
    pub(super) page_readiness_wait_count: u64,
    pub(super) page_readiness_wait_nanoseconds: u64,
    pub(super) maximum_page_readiness_wait_nanoseconds: u64,
    pub(super) page_readiness_failure_count: u64,
    #[cfg(feature = "direct-mlx")]
    pub(super) positional_file_read_metrics:
        Option<Arc<astronomical_runtime_integration::PositionalFileReadMetrics>>,
}

impl ExpertSourceLayerPhaseMeasurement {
    pub(super) fn record_load(
        &mut self,
        logical_source_payload_bytes: u64,
        source_interval_count: u64,
    ) {
        self.logical_source_payload_bytes = self
            .logical_source_payload_bytes
            .saturating_add(logical_source_payload_bytes);
        self.maximum_source_page_payload_bytes = self
            .maximum_source_page_payload_bytes
            .max(logical_source_payload_bytes);
        self.source_interval_count = self
            .source_interval_count
            .saturating_add(source_interval_count);
        self.source_load_count = self.source_load_count.saturating_add(1);
    }

    pub(super) fn record_resident_hit(
        &mut self,
        avoided_source_payload_bytes: u64,
        avoided_source_interval_count: u64,
    ) {
        self.resident_hit_count = self.resident_hit_count.saturating_add(1);
        self.avoided_source_payload_bytes = self
            .avoided_source_payload_bytes
            .saturating_add(avoided_source_payload_bytes);
        self.avoided_source_interval_count = self
            .avoided_source_interval_count
            .saturating_add(avoided_source_interval_count);
    }

    pub(super) fn record_page_readiness_wait(
        &mut self,
        elapsed_nanoseconds: u64,
        did_succeed: bool,
    ) {
        self.page_readiness_wait_count = self.page_readiness_wait_count.saturating_add(1);
        self.page_readiness_wait_nanoseconds = self
            .page_readiness_wait_nanoseconds
            .saturating_add(elapsed_nanoseconds);
        self.maximum_page_readiness_wait_nanoseconds = self
            .maximum_page_readiness_wait_nanoseconds
            .max(elapsed_nanoseconds);
        if !did_succeed {
            self.page_readiness_failure_count = self.page_readiness_failure_count.saturating_add(1);
        }
    }

    pub(super) const fn has_evidence(&self) -> bool {
        self.logical_source_payload_bytes > 0
            || self.source_interval_count > 0
            || self.source_load_count > 0
            || self.resident_hit_count > 0
            || self.avoided_source_payload_bytes > 0
            || self.page_readiness_wait_count > 0
    }

    #[cfg(feature = "direct-mlx")]
    pub(super) fn positional_file_read_metrics(
        &mut self,
    ) -> Arc<astronomical_runtime_integration::PositionalFileReadMetrics> {
        Arc::clone(self.positional_file_read_metrics.get_or_insert_with(|| {
            Arc::new(astronomical_runtime_integration::PositionalFileReadMetrics::default())
        }))
    }
}

pub(crate) type ExpertSourceLayerMeasurement = [ExpertSourceLayerPhaseMeasurement; 3];

pub(super) fn empty_expert_source_layer_measurement() -> ExpertSourceLayerMeasurement {
    std::array::from_fn(|_| ExpertSourceLayerPhaseMeasurement::default())
}
