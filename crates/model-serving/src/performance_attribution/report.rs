use serde::Serialize;

use super::{
    EnabledPerformanceAttribution, PerformanceAttribution, PerformanceCounter,
    PerformanceOperation, unix_epoch_millis,
};

/// Outcome recorded when one bounded attribution report ends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceAttributionOutcome {
    Success,
    Rejected,
    Cancelled,
    Failed,
}

/// Immutable metadata supplied when a model-loading report finishes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoadingPerformanceAttributionMetadata {
    pub outcome: PerformanceAttributionOutcome,
    pub model_id: Option<String>,
    pub model_revision: Option<String>,
    pub prefill_transient_observation_completed: bool,
    pub prefill_observed_transient_high_water_bytes: u64,
    pub retained_complete_expert_layer_count: u64,
    pub total_artifact_payload_bytes: Option<u64>,
    pub resident_model_payload_bytes: Option<u64>,
    pub model_shard_count: Option<usize>,
    pub mlx_active_memory_bytes: Option<u64>,
    pub mlx_allocator_cache_memory_bytes: Option<u64>,
    pub mlx_peak_memory_bytes: Option<u64>,
    pub failure_description: Option<String>,
}

/// Immutable metadata supplied when a generation report finishes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationPerformanceAttributionMetadata {
    pub outcome: PerformanceAttributionOutcome,
    pub model_id: String,
    pub model_revision: String,
    pub prefill_transient_observation_completed: bool,
    pub prefill_observed_transient_high_water_bytes: u64,
    pub retained_complete_expert_layer_count: u64,
    pub request_id: u64,
    pub configured_maximum_output_tokens: u16,
    pub mlx_active_memory_bytes: Option<u64>,
    pub mlx_allocator_cache_memory_bytes: Option<u64>,
    pub mlx_peak_memory_bytes: Option<u64>,
    pub failure_description: Option<String>,
}

/// One serialized attribution report.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "report_kind", rename_all = "snake_case")]
pub enum PerformanceAttributionReport {
    ModelLoading(ModelLoadingPerformanceAttributionReport),
    Generation(GenerationPerformanceAttributionReport),
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelLoadingPerformanceAttributionReport {
    #[serde(flatten)]
    common: CommonPerformanceAttributionReport,
    model_id: Option<String>,
    model_revision: Option<String>,
    prefill_transient_observation_completed: bool,
    prefill_observed_transient_high_water_bytes: u64,
    retained_complete_expert_layer_count: u64,
    total_artifact_payload_bytes: Option<u64>,
    resident_model_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
    mlx_active_memory_bytes: Option<u64>,
    mlx_allocator_cache_memory_bytes: Option<u64>,
    mlx_peak_memory_bytes: Option<u64>,
    failure_description: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenerationPerformanceAttributionReport {
    #[serde(flatten)]
    common: CommonPerformanceAttributionReport,
    model_id: String,
    model_revision: String,
    prefill_transient_observation_completed: bool,
    prefill_observed_transient_high_water_bytes: u64,
    retained_complete_expert_layer_count: u64,
    request_id: u64,
    configured_maximum_output_tokens: u16,
    mlx_active_memory_bytes: Option<u64>,
    mlx_allocator_cache_memory_bytes: Option<u64>,
    mlx_peak_memory_bytes: Option<u64>,
    failure_description: Option<String>,
    previous_token_expert_route_reuse_by_layer: Vec<PreviousTokenExpertRouteReuseByLayerReport>,
}

#[derive(Clone, Debug, Serialize)]
struct PreviousTokenExpertRouteReuseByLayerReport {
    layer_index: usize,
    predicted_expert_count: u64,
    matched_expert_count: u64,
    completely_matched_layer_count: u64,
    examined_layer_count: u64,
}

#[derive(Clone, Debug, Serialize)]
struct CommonPerformanceAttributionReport {
    started_at_unix_millis: u64,
    ended_at_unix_millis: u64,
    report_elapsed_nanoseconds: u64,
    attributed_elapsed_nanoseconds: u64,
    unattributed_elapsed_nanoseconds: u64,
    attributed_percent: f64,
    outcome: PerformanceAttributionOutcome,
    operations: Vec<PerformanceOperationReport>,
    counters: Vec<PerformanceCounterReport>,
}

#[derive(Clone, Debug, Serialize)]
struct PerformanceOperationReport {
    operation: &'static str,
    occurrence_count: u64,
    total_elapsed_nanoseconds: u64,
    minimum_elapsed_nanoseconds: u64,
    maximum_elapsed_nanoseconds: u64,
    first_started_offset_nanoseconds: u64,
    last_ended_offset_nanoseconds: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PerformanceCounterReport {
    counter: &'static str,
    amount: u64,
}

impl PerformanceAttribution {
    #[must_use]
    pub fn finish_model_loading(
        self,
        model_loading_metadata: ModelLoadingPerformanceAttributionMetadata,
    ) -> Option<PerformanceAttributionReport> {
        let enabled_attribution = self.enabled_attribution?;
        Some(PerformanceAttributionReport::ModelLoading(
            ModelLoadingPerformanceAttributionReport {
                common: enabled_attribution.finish_common_report(model_loading_metadata.outcome),
                model_id: model_loading_metadata.model_id,
                model_revision: model_loading_metadata.model_revision,
                prefill_transient_observation_completed: model_loading_metadata
                    .prefill_transient_observation_completed,
                prefill_observed_transient_high_water_bytes: model_loading_metadata
                    .prefill_observed_transient_high_water_bytes,
                retained_complete_expert_layer_count: model_loading_metadata
                    .retained_complete_expert_layer_count,
                total_artifact_payload_bytes: model_loading_metadata.total_artifact_payload_bytes,
                resident_model_payload_bytes: model_loading_metadata.resident_model_payload_bytes,
                model_shard_count: model_loading_metadata.model_shard_count,
                mlx_active_memory_bytes: model_loading_metadata.mlx_active_memory_bytes,
                mlx_allocator_cache_memory_bytes: model_loading_metadata
                    .mlx_allocator_cache_memory_bytes,
                mlx_peak_memory_bytes: model_loading_metadata.mlx_peak_memory_bytes,
                failure_description: model_loading_metadata.failure_description,
            },
        ))
    }

    #[must_use]
    pub fn finish_generation(
        self,
        generation_metadata: GenerationPerformanceAttributionMetadata,
    ) -> Option<PerformanceAttributionReport> {
        let enabled_attribution = self.enabled_attribution?;
        Some(PerformanceAttributionReport::Generation(
            GenerationPerformanceAttributionReport {
                common: enabled_attribution.finish_common_report(generation_metadata.outcome),
                model_id: generation_metadata.model_id,
                model_revision: generation_metadata.model_revision,
                prefill_transient_observation_completed: generation_metadata
                    .prefill_transient_observation_completed,
                prefill_observed_transient_high_water_bytes: generation_metadata
                    .prefill_observed_transient_high_water_bytes,
                retained_complete_expert_layer_count: generation_metadata
                    .retained_complete_expert_layer_count,
                request_id: generation_metadata.request_id,
                configured_maximum_output_tokens: generation_metadata
                    .configured_maximum_output_tokens,
                mlx_active_memory_bytes: generation_metadata.mlx_active_memory_bytes,
                mlx_allocator_cache_memory_bytes: generation_metadata
                    .mlx_allocator_cache_memory_bytes,
                mlx_peak_memory_bytes: generation_metadata.mlx_peak_memory_bytes,
                failure_description: generation_metadata.failure_description,
                previous_token_expert_route_reuse_by_layer: enabled_attribution
                    .previous_token_expert_route_reuse_by_layer
                    .iter()
                    .enumerate()
                    .filter(|(_layer_index, layer_measurement)| {
                        layer_measurement.examined_layer_count > 0
                    })
                    .map(|(layer_index, layer_measurement)| {
                        PreviousTokenExpertRouteReuseByLayerReport {
                            layer_index,
                            predicted_expert_count: layer_measurement.predicted_expert_count,
                            matched_expert_count: layer_measurement.matched_expert_count,
                            completely_matched_layer_count: layer_measurement
                                .completely_matched_layer_count,
                            examined_layer_count: layer_measurement.examined_layer_count,
                        }
                    })
                    .collect(),
            },
        ))
    }
}

impl EnabledPerformanceAttribution {
    fn finish_common_report(
        &self,
        outcome: PerformanceAttributionOutcome,
    ) -> CommonPerformanceAttributionReport {
        let report_elapsed_nanoseconds =
            duration_nanoseconds_saturating(self.report_started_at.elapsed());
        let mut operation_reports = Vec::new();
        let mut attributed_elapsed_nanoseconds = 0_u64;
        for (performance_operation, operation_measurement) in PerformanceOperation::ALL
            .into_iter()
            .zip(self.operation_measurements)
        {
            if operation_measurement.occurrence_count == 0 {
                continue;
            }
            if performance_operation.contributes_to_attributed_elapsed() {
                attributed_elapsed_nanoseconds = attributed_elapsed_nanoseconds
                    .saturating_add(operation_measurement.total_elapsed_nanoseconds);
            }
            operation_reports.push(PerformanceOperationReport {
                operation: performance_operation.identifier(),
                occurrence_count: operation_measurement.occurrence_count,
                total_elapsed_nanoseconds: operation_measurement.total_elapsed_nanoseconds,
                minimum_elapsed_nanoseconds: operation_measurement.minimum_elapsed_nanoseconds,
                maximum_elapsed_nanoseconds: operation_measurement.maximum_elapsed_nanoseconds,
                first_started_offset_nanoseconds: operation_measurement
                    .first_started_offset_nanoseconds,
                last_ended_offset_nanoseconds: operation_measurement.last_ended_offset_nanoseconds,
            });
        }
        #[cfg(feature = "direct-mlx")]
        let counter_values = {
            let mut counter_values = self.counter_values;
            let expert_ssd_read_snapshot = self.expert_ssd_read_metrics.snapshot();
            add_counter_amount(
                &mut counter_values,
                PerformanceCounter::ExpertSsdReadCallCount,
                expert_ssd_read_snapshot.read_call_count,
            );
            add_counter_amount(
                &mut counter_values,
                PerformanceCounter::ExpertSsdReadByteCount,
                expert_ssd_read_snapshot.read_byte_count,
            );
            add_counter_amount(
                &mut counter_values,
                PerformanceCounter::ExpertSsdReadElapsedNanoseconds,
                expert_ssd_read_snapshot.total_read_elapsed_nanoseconds,
            );
            counter_values[PerformanceCounter::ExpertSsdReadMaximumElapsedNanoseconds as usize] =
                counter_values[PerformanceCounter::ExpertSsdReadMaximumElapsedNanoseconds as usize]
                    .max(expert_ssd_read_snapshot.maximum_read_elapsed_nanoseconds);
            add_counter_amount(
                &mut counter_values,
                PerformanceCounter::ExpertSsdReadFailureCount,
                expert_ssd_read_snapshot.read_failure_count,
            );
            counter_values
        };
        #[cfg(not(feature = "direct-mlx"))]
        let counter_values = self.counter_values;
        let mut counter_reports = Vec::new();
        for (performance_counter, counter_amount) in
            PerformanceCounter::ALL.into_iter().zip(counter_values)
        {
            if counter_amount == 0 {
                continue;
            }
            counter_reports.push(PerformanceCounterReport {
                counter: performance_counter.identifier(),
                amount: counter_amount,
            });
        }
        let unattributed_elapsed_nanoseconds =
            report_elapsed_nanoseconds.saturating_sub(attributed_elapsed_nanoseconds);
        let attributed_percent = if report_elapsed_nanoseconds == 0 {
            0.0
        } else {
            attributed_elapsed_nanoseconds as f64 / report_elapsed_nanoseconds as f64 * 100.0
        };
        CommonPerformanceAttributionReport {
            started_at_unix_millis: self.report_started_at_unix_millis,
            ended_at_unix_millis: unix_epoch_millis(),
            report_elapsed_nanoseconds,
            attributed_elapsed_nanoseconds,
            unattributed_elapsed_nanoseconds,
            attributed_percent,
            outcome,
            operations: operation_reports,
            counters: counter_reports,
        }
    }
}

#[cfg(feature = "direct-mlx")]
fn add_counter_amount(
    counter_values: &mut [u64; PerformanceCounter::COUNT],
    performance_counter: PerformanceCounter,
    amount: u64,
) {
    counter_values[performance_counter as usize] =
        counter_values[performance_counter as usize].saturating_add(amount);
}

fn duration_nanoseconds_saturating(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
