//! Switchable, request-owned attribution with a zero-allocation disabled path.
//!
//! Enabled reports aggregate fixed operation and counter arrays, avoiding maps
//! and per-event records on inference paths. Disabled reports hold only `None`,
//! skip clock reads, and execute measured closures directly.

mod catalog;
mod counter_catalog;
mod expert_source;
mod log;
mod measurement_catalog;
mod report;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "direct-mlx")]
use std::sync::Arc;

pub use catalog::PerformanceOperation;
pub use counter_catalog::PerformanceCounter;
pub use expert_source::ExpertSourceRequestPhase;
pub use log::PerformanceAttributionLog;
pub use measurement_catalog::PerformanceOperationMeasurement;
pub use report::{
    GenerationPerformanceAttributionMetadata, ModelLoadingPerformanceAttributionMetadata,
    PerformanceAttributionOutcome, PerformanceAttributionReport,
};

/// Pointer-sized disabled handle with a fixed-size enabled accumulator.
#[derive(Clone, Debug)]
pub struct PerformanceAttribution {
    pub(super) enabled_attribution: Option<Box<EnabledPerformanceAttribution>>,
}

#[derive(Clone, Debug)]
pub(super) struct EnabledPerformanceAttribution {
    pub(super) report_started_at: Instant,
    pub(super) report_started_at_unix_millis: u64,
    pub(super) operation_measurements:
        [PerformanceOperationMeasurement; PerformanceOperation::COUNT],
    pub(super) counter_values: [u64; PerformanceCounter::COUNT],
    pub(super) previous_token_selected_expert_ids_by_layer: Vec<Option<Vec<usize>>>,
    pub(super) previous_token_expert_route_reuse_by_layer:
        Vec<PreviousTokenExpertRouteReuseMeasurement>,
    pub(super) expert_source_by_layer: Vec<expert_source::ExpertSourceLayerMeasurement>,
    #[cfg(feature = "direct-mlx")]
    pub(super) positional_file_read_metrics:
        Arc<astronomical_runtime_integration::PositionalFileReadMetrics>,
    #[cfg(feature = "direct-mlx")]
    // Capture cumulative process I/O at the same boundary as the monotonic report
    // clock. Keeping the Result preserves sampling failure as explicit evidence;
    // replacing failure with zero would falsely claim that macOS served no disk I/O.
    pub(super) process_io_start: Result<
        astronomical_runtime_integration::MacosProcessIoSnapshot,
        astronomical_runtime_integration::MacosProcessIoError,
    >,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PreviousTokenExpertRouteReuseMeasurement {
    pub(super) predicted_expert_count: u64,
    pub(super) matched_expert_count: u64,
    pub(super) completely_matched_layer_count: u64,
    pub(super) examined_layer_count: u64,
}

impl PerformanceAttribution {
    /// Creates a no-op accumulator that performs no clock reads.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled_attribution: None,
        }
    }

    /// Creates an enabled accumulator and captures its monotonic start boundary.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled_attribution: Some(Box::new(EnabledPerformanceAttribution {
                report_started_at: Instant::now(),
                report_started_at_unix_millis: unix_epoch_millis(),
                operation_measurements: [PerformanceOperationMeasurement::EMPTY;
                    PerformanceOperation::COUNT],
                counter_values: [0; PerformanceCounter::COUNT],
                previous_token_selected_expert_ids_by_layer: Vec::new(),
                previous_token_expert_route_reuse_by_layer: Vec::new(),
                expert_source_by_layer: Vec::new(),
                #[cfg(feature = "direct-mlx")]
                positional_file_read_metrics: Arc::new(
                    astronomical_runtime_integration::PositionalFileReadMetrics::default(),
                ),
                #[cfg(feature = "direct-mlx")]
                process_io_start: astronomical_runtime_integration::sample_current_process_io(),
            })),
        }
    }

    /// Measures one operation, including error-returning operations, when enabled.
    pub fn measure_operation<OperationOutput>(
        &mut self,
        operation: PerformanceOperation,
        measured_operation: impl FnOnce(&mut Self) -> OperationOutput,
    ) -> OperationOutput {
        let report_started_at = match self.enabled_attribution.as_ref() {
            Some(enabled_attribution) => enabled_attribution.report_started_at,
            None => return measured_operation(self),
        };
        let operation_started_at = Instant::now();
        let operation_output = measured_operation(self);
        let operation_ended_at = Instant::now();
        self.record_completed_operation(
            operation,
            operation_started_at.saturating_duration_since(report_started_at),
            operation_ended_at.saturating_duration_since(report_started_at),
        );
        operation_output
    }

    /// Starts an outer diagnostic span without disabling nested measurements.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn begin_operation_span(&self) -> Option<Instant> {
        self.enabled_attribution.as_ref().map(|_| Instant::now())
    }

    /// Completes an outer diagnostic span started by `begin_operation_span`.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn complete_operation_span(
        &mut self,
        operation: PerformanceOperation,
        operation_started_at: Option<Instant>,
    ) {
        let (Some(enabled_attribution), Some(operation_started_at)) =
            (self.enabled_attribution.as_ref(), operation_started_at)
        else {
            return;
        };
        let report_started_at = enabled_attribution.report_started_at;
        self.record_completed_operation(
            operation,
            operation_started_at.saturating_duration_since(report_started_at),
            Instant::now().saturating_duration_since(report_started_at),
        );
    }

    /// Returns whether this accumulator records measurements.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled_attribution.is_some()
    }

    #[cfg(feature = "direct-mlx")]
    pub(crate) fn positional_file_read_metrics(
        &self,
    ) -> Option<Arc<astronomical_runtime_integration::PositionalFileReadMetrics>> {
        self.enabled_attribution
            .as_ref()
            .map(|enabled_attribution| {
                Arc::clone(&enabled_attribution.positional_file_read_metrics)
            })
    }

    /// Records deterministic offsets; exposed for aggregation tests and pre-measured boundaries.
    pub fn record_completed_operation(
        &mut self,
        operation: PerformanceOperation,
        started_offset: Duration,
        ended_offset: Duration,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        enabled_attribution.operation_measurements[operation as usize]
            .record(started_offset, ended_offset);
    }

    /// Returns the nonempty aggregate for one operation.
    #[must_use]
    pub fn operation_measurement(
        &self,
        operation: PerformanceOperation,
    ) -> Option<PerformanceOperationMeasurement> {
        let enabled_attribution = self.enabled_attribution.as_ref()?;
        let operation_measurement = enabled_attribution.operation_measurements[operation as usize];
        (operation_measurement.occurrence_count > 0).then_some(operation_measurement)
    }

    /// Adds a bounded amount to one report counter.
    pub fn record_counter(&mut self, counter: PerformanceCounter, amount: u64) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        enabled_attribution.counter_values[counter as usize] =
            enabled_attribution.counter_values[counter as usize].saturating_add(amount);
    }

    /// Retains the largest observed amount for one report counter.
    pub fn record_maximum_counter(&mut self, counter: PerformanceCounter, amount: u64) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        enabled_attribution.counter_values[counter as usize] =
            enabled_attribution.counter_values[counter as usize].max(amount);
    }

    /// Aggregates one expert source request without retaining per-event records.
    pub fn record_expert_source_load(
        &mut self,
        layer_index: usize,
        request_phase: ExpertSourceRequestPhase,
        logical_source_payload_bytes: u64,
        source_interval_count: u64,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        ensure_expert_source_layer(enabled_attribution, layer_index);
        enabled_attribution.expert_source_by_layer[layer_index][request_phase.index()]
            .record_load(logical_source_payload_bytes, source_interval_count);
    }

    /// Records that a retained complete layer avoided a source request.
    pub fn record_expert_source_resident_hit(
        &mut self,
        layer_index: usize,
        request_phase: ExpertSourceRequestPhase,
        avoided_source_payload_bytes: u64,
        avoided_source_interval_count: u64,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        ensure_expert_source_layer(enabled_attribution, layer_index);
        enabled_attribution.expert_source_by_layer[layer_index][request_phase.index()]
            .record_resident_hit(avoided_source_payload_bytes, avoided_source_interval_count);
    }

    /// Records one explicit page-materialization wait using caller-supplied timing.
    pub fn record_expert_page_readiness_wait(
        &mut self,
        layer_index: usize,
        request_phase: ExpertSourceRequestPhase,
        elapsed: Duration,
        did_succeed: bool,
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        ensure_expert_source_layer(enabled_attribution, layer_index);
        enabled_attribution.expert_source_by_layer[layer_index][request_phase.index()]
            .record_page_readiness_wait(
                u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
                did_succeed,
            );
    }

    /// Materializes one expert page and attributes the blocking readiness boundary.
    pub fn measure_expert_page_readiness<ReadinessOutput, ReadinessError>(
        &mut self,
        layer_index: usize,
        request_phase: ExpertSourceRequestPhase,
        materialize_page: impl FnOnce() -> Result<ReadinessOutput, ReadinessError>,
    ) -> Result<ReadinessOutput, ReadinessError> {
        if self.enabled_attribution.is_none() {
            return materialize_page();
        }
        let readiness_started_at = Instant::now();
        let readiness_outcome = materialize_page();
        self.record_expert_page_readiness_wait(
            layer_index,
            request_phase,
            readiness_started_at.elapsed(),
            readiness_outcome.is_ok(),
        );
        readiness_outcome
    }

    /// Returns the phase/layer read accumulator used by bounded lazy MLX reads.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn expert_source_positional_file_read_metrics(
        &mut self,
        layer_index: usize,
        request_phase: ExpertSourceRequestPhase,
    ) -> Option<Arc<astronomical_runtime_integration::PositionalFileReadMetrics>> {
        let enabled_attribution = self.enabled_attribution.as_mut()?;
        ensure_expert_source_layer(enabled_attribution, layer_index);
        Some(
            enabled_attribution.expert_source_by_layer[layer_index][request_phase.index()]
                .positional_file_read_metrics(),
        )
    }

    /// Compares one decode layer's selected experts with the preceding decode token.
    pub fn record_previous_token_expert_route_reuse(
        &mut self,
        layer_index: usize,
        token_count: i32,
        selected_expert_ids: &[usize],
    ) {
        let Some(enabled_attribution) = self.enabled_attribution.as_mut() else {
            return;
        };
        if token_count != 1 {
            return;
        }

        let mut sorted_unique_selected_expert_ids = selected_expert_ids.to_vec();
        sorted_unique_selected_expert_ids.sort_unstable();
        sorted_unique_selected_expert_ids.dedup();
        if enabled_attribution
            .previous_token_selected_expert_ids_by_layer
            .len()
            <= layer_index
        {
            enabled_attribution
                .previous_token_selected_expert_ids_by_layer
                .resize_with(layer_index + 1, || None);
        }
        let previous_selected_expert_ids = enabled_attribution
            .previous_token_selected_expert_ids_by_layer[layer_index]
            .replace(sorted_unique_selected_expert_ids.clone());
        let Some(previous_selected_expert_ids) = previous_selected_expert_ids else {
            return;
        };

        let predicted_expert_count = usize_to_u64_saturating(previous_selected_expert_ids.len());
        let matched_expert_count = usize_to_u64_saturating(
            previous_selected_expert_ids
                .iter()
                .filter(|expert_id| {
                    sorted_unique_selected_expert_ids
                        .binary_search(expert_id)
                        .is_ok()
                })
                .count(),
        );
        let completely_matched_layer_count =
            u64::from(previous_selected_expert_ids == sorted_unique_selected_expert_ids);
        if enabled_attribution
            .previous_token_expert_route_reuse_by_layer
            .len()
            <= layer_index
        {
            enabled_attribution
                .previous_token_expert_route_reuse_by_layer
                .resize(
                    layer_index + 1,
                    PreviousTokenExpertRouteReuseMeasurement::default(),
                );
        }
        let layer_measurement =
            &mut enabled_attribution.previous_token_expert_route_reuse_by_layer[layer_index];
        layer_measurement.predicted_expert_count = layer_measurement
            .predicted_expert_count
            .saturating_add(predicted_expert_count);
        layer_measurement.matched_expert_count = layer_measurement
            .matched_expert_count
            .saturating_add(matched_expert_count);
        layer_measurement.completely_matched_layer_count = layer_measurement
            .completely_matched_layer_count
            .saturating_add(completely_matched_layer_count);
        layer_measurement.examined_layer_count =
            layer_measurement.examined_layer_count.saturating_add(1);

        for (performance_counter, counter_increment) in [
            (
                PerformanceCounter::ExpertRoutePredictedExpertCount,
                predicted_expert_count,
            ),
            (
                PerformanceCounter::ExpertRouteMatchedExpertCount,
                matched_expert_count,
            ),
            (
                PerformanceCounter::ExpertRouteCompletelyMatchedLayerCount,
                completely_matched_layer_count,
            ),
            (PerformanceCounter::ExpertRouteExaminedLayerCount, 1),
        ] {
            enabled_attribution.counter_values[performance_counter as usize] = enabled_attribution
                .counter_values[performance_counter as usize]
                .saturating_add(counter_increment);
        }
    }

    /// Returns one accumulated counter, or zero when attribution is disabled.
    #[must_use]
    pub fn counter_value(&self, counter: PerformanceCounter) -> u64 {
        self.enabled_attribution
            .as_ref()
            .map_or(0, |enabled_attribution| {
                enabled_attribution.counter_values[counter as usize]
            })
    }
}

fn ensure_expert_source_layer(
    enabled_attribution: &mut EnabledPerformanceAttribution,
    layer_index: usize,
) {
    if enabled_attribution.expert_source_by_layer.len() <= layer_index {
        enabled_attribution.expert_source_by_layer.resize_with(
            layer_index.saturating_add(1),
            expert_source::empty_expert_source_layer_measurement,
        );
    }
}

fn usize_to_u64_saturating(integer_count: usize) -> u64 {
    u64::try_from(integer_count).map_or(u64::MAX, |converted_count| converted_count)
}

pub(super) fn unix_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration_since_epoch| {
            u64::try_from(duration_since_epoch.as_millis()).unwrap_or(u64::MAX)
        })
}
