use std::time::Instant;

use crate::{
    GenerationFinalization, GenerationPerformanceAttributionMetadata, InferenceEngineError,
    MlxMemoryTelemetry, PerformanceAttribution, PerformanceAttributionOutcome, PerformanceCounter,
    PerformanceOperation,
};

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;

impl Qwen3_5EngineState {
    pub(super) fn collect_current_mlx_memory_telemetry(
        &self,
    ) -> Result<Option<MlxMemoryTelemetry>, InferenceEngineError> {
        let Some(model) = self.model.as_ref() else {
            return Ok(None);
        };
        let mlx_memory_snapshot = model
            .runtime()
            .memory_snapshot()
            .map_err(super::qwen3_5_runtime_error)?;
        let active_memory_bytes = u64::try_from(mlx_memory_snapshot.active_memory_bytes())
            .map_err(|_| {
                super::fatal_engine_error("MLX active memory bytes exceed the u64 range")
            })?;
        let allocator_cache_memory_bytes =
            u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()).map_err(|_| {
                super::fatal_engine_error("MLX allocator-cache memory bytes exceed the u64 range")
            })?;
        let peak_memory_bytes = u64::try_from(mlx_memory_snapshot.peak_memory_bytes())
            .map_err(|_| super::fatal_engine_error("MLX peak memory bytes exceed the u64 range"))?;
        Ok(Some(MlxMemoryTelemetry::new(
            active_memory_bytes,
            allocator_cache_memory_bytes,
            peak_memory_bytes,
            model.finalized_active_memory_breakdown(active_memory_bytes, 0),
        )))
    }

    pub(super) fn finalize_generation_request_after_error(
        &mut self,
        active_request: Qwen3_5EngineRequest,
        generation_error: &InferenceEngineError,
        rejection_description: &'static str,
        failure_description: &'static str,
    ) {
        let (performance_attribution_outcome, performance_attribution_failure_description) =
            match generation_error {
                InferenceEngineError::InvalidRequest { .. }
                | InferenceEngineError::MlxMemoryLimitRejected { .. }
                | InferenceEngineError::EngineBusy => (
                    PerformanceAttributionOutcome::Rejected,
                    rejection_description,
                ),
                InferenceEngineError::Fatal { .. } => {
                    (PerformanceAttributionOutcome::Failed, failure_description)
                }
            };
        let _generation_finalization = self.finalize_generation_request(
            active_request,
            performance_attribution_outcome,
            Some(performance_attribution_failure_description),
        );
    }

    pub(super) fn finalize_generation_request(
        &mut self,
        mut active_request: Qwen3_5EngineRequest,
        outcome: PerformanceAttributionOutcome,
        failure_description: Option<&'static str>,
    ) -> GenerationFinalization {
        let request_id = active_request.request_id;
        let configured_maximum_output_tokens = active_request.maximum_output_tokens;
        let expert_weight_memory_cache_statistics_at_request_start =
            active_request.expert_weight_memory_cache_statistics_at_request_start;
        let mut performance_attribution = std::mem::replace(
            &mut active_request.performance_attribution,
            PerformanceAttribution::disabled(),
        );
        drop(active_request);
        let resumed_after_request_memory_pressure = self
            .model
            .as_ref()
            .is_some_and(|model| model.resume_expert_retention_after_request_memory_pressure());
        let should_recover_complete_expert_layers = self.model.as_ref().is_some_and(|model| {
            model.expert_memory_mode() != astronomical_ipc_protocol::ExpertMemoryMode::Resident
        });
        if let Some(model) = self.model.as_ref() {
            let expert_weight_memory_cache_statistics =
                model.expert_weight_memory_cache_statistics();
            tracing::info!(
                request_id = request_id.value(),
                resumed_after_request_memory_pressure,
                expert_memory_mode = ?model.expert_memory_mode(),
                retained_complete_layer_count =
                    expert_weight_memory_cache_statistics.complete_layer_count,
                retained_expert_payload_bytes =
                    expert_weight_memory_cache_statistics.resident_payload_byte_count,
                maximum_retained_expert_payload_bytes =
                    expert_weight_memory_cache_statistics.maximum_resident_payload_byte_count,
                "released request-scoped expert retention ceiling"
            );
        }
        let mlx_memory_snapshot = performance_attribution.measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                self.release_request_memory(
                    request_id,
                    self.adaptive_ram_growth_guard_enabled || should_recover_complete_expert_layers,
                )
            },
        );
        if should_recover_complete_expert_layers && mlx_memory_snapshot.is_none() {
            tracing::warn!(
                request_id = request_id.value(),
                "skipped complete expert-layer recovery because request-memory cleanup was not confirmed"
            );
        }
        if should_recover_complete_expert_layers
            && mlx_memory_snapshot.is_some()
            && let Some(model) = self.model.as_ref()
        {
            let expert_layer_recovery_started_at = Instant::now();
            let expert_weight_memory_cache_statistics_before_recovery =
                model.expert_weight_memory_cache_statistics();
            tracing::info!(
                request_id = request_id.value(),
                retained_complete_layer_count_before =
                    expert_weight_memory_cache_statistics_before_recovery.complete_layer_count,
                retained_expert_payload_bytes_before =
                    expert_weight_memory_cache_statistics_before_recovery
                        .resident_payload_byte_count,
                "started complete expert-layer recovery after request-memory cleanup"
            );
            let expert_layer_recovery_outcome = model
                .prewarm_complete_expert_layers_with_performance_attribution(
                    &mut performance_attribution,
                );
            let expert_weight_memory_cache_statistics_after_recovery =
                model.expert_weight_memory_cache_statistics();
            tracing::info!(
                request_id = request_id.value(),
                recovery_succeeded = expert_layer_recovery_outcome.is_ok(),
                recovery_elapsed_millis =
                    expert_layer_recovery_started_at.elapsed().as_millis(),
                expert_memory_mode_after_recovery = ?model.expert_memory_mode(),
                retained_complete_layer_count_before =
                    expert_weight_memory_cache_statistics_before_recovery.complete_layer_count,
                retained_complete_layer_count_after =
                    expert_weight_memory_cache_statistics_after_recovery.complete_layer_count,
                retained_expert_payload_bytes_before =
                    expert_weight_memory_cache_statistics_before_recovery.resident_payload_byte_count,
                retained_expert_payload_bytes_after =
                    expert_weight_memory_cache_statistics_after_recovery.resident_payload_byte_count,
                maximum_retained_expert_payload_bytes_after =
                    expert_weight_memory_cache_statistics_after_recovery.maximum_resident_payload_byte_count,
                "finished complete expert-layer recovery after request-memory cleanup"
            );
            if let Err(expert_layer_recovery_error) = expert_layer_recovery_outcome {
                tracing::warn!(
                    request_id = request_id.value(),
                    error = %expert_layer_recovery_error,
                    "could not recover complete expert layers after request memory was released"
                );
            }
        }
        if performance_attribution.is_enabled()
            && let Some(model) = self.model.as_ref()
        {
            let expert_weight_memory_cache_statistics_at_request_end =
                model.expert_weight_memory_cache_statistics();
            record_final_expert_weight_memory_cache_counters(
                &mut performance_attribution,
                expert_weight_memory_cache_statistics_at_request_start,
                expert_weight_memory_cache_statistics_at_request_end,
            );
        }
        let generation_finalization =
            self.collect_generation_finalization(&mut performance_attribution);
        self.record_generation_performance_attribution(
            performance_attribution,
            outcome,
            request_id,
            configured_maximum_output_tokens,
            mlx_memory_snapshot,
            failure_description,
        );
        generation_finalization
    }

    fn collect_generation_finalization(
        &self,
        performance_attribution: &mut PerformanceAttribution,
    ) -> GenerationFinalization {
        let Some(model) = self.model.as_ref() else {
            return GenerationFinalization::default();
        };
        let expert_memory_mode = Some(model.expert_memory_mode());
        let mlx_memory_telemetry = match performance_attribution.measure_operation(
            PerformanceOperation::FinalizedMlxMemorySnapshot,
            |_performance_attribution| model.runtime().memory_snapshot(),
        ) {
            Ok(mlx_memory_snapshot) => match (
                u64::try_from(mlx_memory_snapshot.active_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.allocator_cache_memory_bytes()),
                u64::try_from(mlx_memory_snapshot.peak_memory_bytes()),
            ) {
                (
                    Ok(active_memory_bytes),
                    Ok(allocator_cache_memory_bytes),
                    Ok(peak_memory_bytes),
                ) => Some(MlxMemoryTelemetry::new(
                    active_memory_bytes,
                    allocator_cache_memory_bytes,
                    peak_memory_bytes,
                    model.finalized_active_memory_breakdown(active_memory_bytes, 0),
                )),
                memory_counter_results => {
                    tracing::warn!(
                        ?memory_counter_results,
                        "could not publish finalized MLX memory because a runtime counter exceeds the u64 range"
                    );
                    None
                }
            },
            Err(memory_snapshot_error) => {
                tracing::warn!(
                    error = %memory_snapshot_error,
                    "could not capture finalized MLX memory after expert residency recovery"
                );
                None
            }
        };
        GenerationFinalization::new(expert_memory_mode, mlx_memory_telemetry)
    }

    pub(super) fn record_generation_performance_attribution(
        &mut self,
        performance_attribution: PerformanceAttribution,
        outcome: PerformanceAttributionOutcome,
        request_id: astronomical_ipc_protocol::RequestId,
        configured_maximum_output_tokens: u16,
        mlx_memory_snapshot: Option<astronomical_runtime_integration::MlxMemorySnapshot>,
        failure_description: Option<&'static str>,
    ) {
        if !performance_attribution.is_enabled() {
            return;
        }
        let (Some(model_id), Some(model_revision)) =
            (self.model_id.clone(), self.model_revision.clone())
        else {
            tracing::error!(
                request_id = request_id.value(),
                "completed generation lacked loaded-model attribution metadata"
            );
            return;
        };
        let Some(performance_attribution_report) =
            performance_attribution.finish_generation(GenerationPerformanceAttributionMetadata {
                outcome,
                model_id,
                model_revision,
                prefill_transient_observation_completed: self
                    .adaptive_ram_growth_guard
                    .has_completed_growth_observation(crate::AdaptiveRamGrowthPhase::Prefill),
                prefill_observed_transient_high_water_bytes: u64::try_from(
                    self.adaptive_ram_growth_guard
                        .observed_transient_high_water_bytes(
                            crate::AdaptiveRamGrowthPhase::Prefill,
                        ),
                )
                .unwrap_or(u64::MAX),
                retained_complete_expert_layer_count: self.model.as_ref().map_or(0, |model| {
                    u64::try_from(
                        model
                            .expert_weight_memory_cache_statistics()
                            .complete_layer_count,
                    )
                    .unwrap_or(u64::MAX)
                }),
                request_id: request_id.value(),
                configured_maximum_output_tokens,
                mlx_active_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.active_memory_bytes()).ok()),
                mlx_allocator_cache_memory_bytes: mlx_memory_snapshot.as_ref().and_then(
                    |snapshot| u64::try_from(snapshot.allocator_cache_memory_bytes()).ok(),
                ),
                mlx_peak_memory_bytes: mlx_memory_snapshot
                    .as_ref()
                    .and_then(|snapshot| u64::try_from(snapshot.peak_memory_bytes()).ok()),
                failure_description: failure_description.map(str::to_owned),
            })
        else {
            return;
        };
        if let Err(performance_attribution_write_error) = self
            .performance_attribution_log
            .record(&performance_attribution_report)
        {
            tracing::warn!(
                error = %performance_attribution_write_error,
                "failed to append generation performance attribution"
            );
        }
    }
}

fn record_final_expert_weight_memory_cache_counters(
    performance_attribution: &mut PerformanceAttribution,
    expert_weight_memory_cache_statistics_at_request_start:
        crate::expert_paging::ExpertWeightMemoryCacheStatistics,
    expert_weight_memory_cache_statistics_at_request_end:
        crate::expert_paging::ExpertWeightMemoryCacheStatistics,
) {
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightMemoryCacheHitCount,
        expert_weight_memory_cache_statistics_at_request_end
            .cache_hit_count
            .saturating_sub(expert_weight_memory_cache_statistics_at_request_start.cache_hit_count),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightMemoryCacheCompleteLayerHitCount,
        expert_weight_memory_cache_statistics_at_request_end
            .complete_layer_hit_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_at_request_start.complete_layer_hit_count,
            ),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightMemoryCacheMissCount,
        expert_weight_memory_cache_statistics_at_request_end
            .cache_miss_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_at_request_start.cache_miss_count,
            ),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightMemoryCacheEvictionCount,
        expert_weight_memory_cache_statistics_at_request_end
            .eviction_count
            .saturating_sub(expert_weight_memory_cache_statistics_at_request_start.eviction_count),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightDiskPageLoadCount,
        expert_weight_memory_cache_statistics_at_request_end
            .disk_page_load_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_at_request_start.disk_page_load_count,
            ),
    );
    performance_attribution.record_counter(
        PerformanceCounter::ExpertWeightDiskBatchLoadCount,
        expert_weight_memory_cache_statistics_at_request_end
            .disk_batch_load_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_at_request_start.disk_batch_load_count,
            ),
    );
}
