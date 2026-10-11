//! Fixed-shape admission for one resident Qwen3.5 forward.

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_resident::model::adaptive_ram_growth_logging;
use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, MemoryPhase, PerformanceAttribution,
    PerformanceCounter, PerformanceOperation, combined_persistent_growth_bytes,
};

use super::Qwen3_5EngineState;
use super::qwen3_5_runtime_error;

pub(in crate::qwen3_5_resident) enum AdaptiveRamGrowthMemoryAdmissionError {
    InsufficientCapacity { reason: String },
    Engine(InferenceEngineError),
}

impl From<InferenceEngineError> for AdaptiveRamGrowthMemoryAdmissionError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for InferenceEngineError {
    fn from(admission_error: AdaptiveRamGrowthMemoryAdmissionError) -> Self {
        match admission_error {
            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason } => {
                crate::qwen3_5_resident::model::memory_admission::invalid_request_error(reason)
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                inference_engine_error
            }
        }
    }
}

pub(in crate::qwen3_5_resident) struct AdaptiveRamGrowthAdmissionBaseline {
    pub(in crate::qwen3_5_resident) active_memory_bytes: usize,
    pub(in crate::qwen3_5_resident) retained_expert_payload_bytes: u64,
    pub(in crate::qwen3_5_resident) streamed_expert_page_bytes: u64,
    pub(in crate::qwen3_5_resident) transient_reserve_source:
        Option<crate::AdaptiveRamGrowthTransientReserveSource>,
}

impl AdaptiveRamGrowthAdmissionBaseline {
    pub(in crate::qwen3_5_resident) const fn disabled_guard_sentinel() -> Self {
        Self {
            active_memory_bytes: usize::MAX,
            retained_expert_payload_bytes: u64::MAX,
            streamed_expert_page_bytes: 0,
            transient_reserve_source: None,
        }
    }
}

impl Qwen3_5EngineState {
    pub(in crate::qwen3_5_resident) fn measure_adaptive_ram_growth_memory_admission(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        performance_attribution: &mut PerformanceAttribution,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
    ) -> Result<AdaptiveRamGrowthAdmissionBaseline, AdaptiveRamGrowthMemoryAdmissionError> {
        if !self.adaptive_ram_growth_guard_enabled {
            return Ok(AdaptiveRamGrowthAdmissionBaseline::disabled_guard_sentinel());
        }
        let admission_performance_attribution_enabled = performance_attribution.is_enabled();
        let admission_baseline = performance_attribution.measure_operation(
            PerformanceOperation::AdaptiveRamGrowthMemoryAdmission,
            |performance_attribution| {
                self.begin_adaptive_ram_growth(
                    adaptive_ram_growth_context,
                    request_decoder_state,
                    additional_persistent_state_growth_bytes,
                    exact_temporary_workspace_bytes,
                    admission_performance_attribution_enabled,
                    performance_attribution,
                )
            },
        )?;
        if let Some(transient_reserve_source) = admission_baseline.transient_reserve_source {
            let admission_reserve_source_counter = match transient_reserve_source {
                crate::AdaptiveRamGrowthTransientReserveSource::ExactContext => {
                    PerformanceCounter::AdmissionReserveExactContextSourceCount
                }
                crate::AdaptiveRamGrowthTransientReserveSource::PhaseScaled => {
                    PerformanceCounter::AdmissionReservePhaseScaledSourceCount
                }
                crate::AdaptiveRamGrowthTransientReserveSource::GlobalMaximum => {
                    PerformanceCounter::AdmissionReserveGlobalMaximumSourceCount
                }
            };
            performance_attribution.record_counter(admission_reserve_source_counter, 1);
        }
        Ok(admission_baseline)
    }

    fn begin_adaptive_ram_growth(
        &mut self,
        adaptive_ram_growth_context: AdaptiveRamGrowthContext,
        request_decoder_state: &RequestDecoderStateStack,
        additional_persistent_state_growth_bytes: usize,
        exact_temporary_workspace_bytes: usize,
        should_log_memory_decision: bool,
        _performance_attribution: &PerformanceAttribution,
    ) -> Result<AdaptiveRamGrowthAdmissionBaseline, AdaptiveRamGrowthMemoryAdmissionError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let target_persistent_state_growth_bytes = request_decoder_state
            .projected_persistent_state_growth_bytes(
                model.decoder_cache_layout(),
                adaptive_ram_growth_context.forward_token_count(),
            )
            .map_err(qwen3_5_runtime_error)?;
        let exact_context_growth_bytes = combined_persistent_growth_bytes(
            target_persistent_state_growth_bytes,
            additional_persistent_state_growth_bytes,
        )
        .ok_or_else(|| {
            crate::qwen3_5_resident::model::memory_admission::invalid_request_error(
                "target and additional persistent growth overflowed",
            )
        })?;
        let memory_snapshot_before_growth = model
            .runtime()
            .memory_snapshot()
            .map_err(qwen3_5_runtime_error)?;
        let forward_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_context_growth_bytes,
                0,
                exact_temporary_workspace_bytes,
            )
            .map_err(|projection_error| {
                crate::qwen3_5_resident::model::memory_admission::invalid_request_error(format!(
                    "adaptive RAM growth rejected: {projection_error}"
                ))
            })?;
        if should_log_memory_decision {
            adaptive_ram_growth_logging::log_adaptive_ram_growth_admission_decision(
                adaptive_ram_growth_context,
                &forward_projection,
                if forward_projection.fits_stable_and_peak_limits() {
                    "admit_resident_forward"
                } else {
                    "request_streaming_engine"
                },
            );
        }
        if !forward_projection.fits_stable_and_peak_limits() {
            return Err(
                AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity {
                    reason: format!(
                        "resident forward requires {} bytes at stable projection and {} bytes at peak projection; allowed ceiling is {} bytes",
                        forward_projection.stable_projected_bytes(),
                        forward_projection.peak_projected_bytes(),
                        forward_projection.active_memory_ceiling_bytes(),
                    ),
                },
            );
        }
        model
            .runtime()
            .reset_peak_memory()
            .map_err(qwen3_5_runtime_error)?;
        let resident_payload_bytes = model
            .resident_expert_weights
            .as_ref()
            .map_or(0, |resident_expert_weights| {
                resident_expert_weights.payload_byte_count()
            });
        Ok(AdaptiveRamGrowthAdmissionBaseline {
            active_memory_bytes: memory_snapshot_before_growth.active_memory_bytes(),
            retained_expert_payload_bytes: resident_payload_bytes,
            streamed_expert_page_bytes: performance_attribution_payload_bytes(
                _performance_attribution,
            ),
            transient_reserve_source: Some(forward_projection.transient_reserve_source()),
        })
    }
}

fn performance_attribution_payload_bytes(performance_attribution: &PerformanceAttribution) -> u64 {
    performance_attribution.expert_streaming_payload_byte_count()
}

pub(crate) fn resident_prefill_operation_token_count(
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
) -> Option<usize> {
    if adaptive_ram_growth_context.memory_phase() == MemoryPhase::Prefill {
        Some(adaptive_ram_growth_context.forward_token_count())
    } else {
        None
    }
}
