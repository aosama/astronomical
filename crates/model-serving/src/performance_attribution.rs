mod catalog;
mod log;
mod report;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "direct-mlx")]
use std::sync::Arc;

pub use catalog::{PerformanceCounter, PerformanceOperation, PerformanceOperationMeasurement};
pub use log::PerformanceAttributionLog;
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
    #[cfg(feature = "direct-mlx")]
    pub(super) expert_ssd_read_metrics: Arc<astronomical_runtime_integration::ExpertSsdReadMetrics>,
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
                #[cfg(feature = "direct-mlx")]
                expert_ssd_read_metrics: Arc::new(
                    astronomical_runtime_integration::ExpertSsdReadMetrics::default(),
                ),
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
    pub(crate) fn expert_ssd_read_metrics(
        &self,
    ) -> Option<Arc<astronomical_runtime_integration::ExpertSsdReadMetrics>> {
        self.enabled_attribution
            .as_ref()
            .map(|enabled_attribution| Arc::clone(&enabled_attribution.expert_ssd_read_metrics))
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
