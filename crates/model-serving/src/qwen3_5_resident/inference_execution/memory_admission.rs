//! Fixed-shape admission for one resident Qwen3.5 forward.

use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_resident::model::adaptive_ram_growth_logging;
use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthProjection, InferenceEngineError, MemoryPhase,
    PerformanceAttribution, PerformanceCounter, PerformanceOperation,
    ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection,
    StreamingEngineRequirement, combined_persistent_growth_bytes,
};

use super::Qwen3_5EngineState;
use super::qwen3_5_runtime_error;

pub(in crate::qwen3_5_resident) enum AdaptiveRamGrowthMemoryAdmissionError {
    /// The typed policy rejection the resident caller must enact.
    ForwardAdmission {
        rejection: ResidentForwardAdmissionRejection,
        reason: String,
    },
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
            // Prefill chunk attempts intercept the typed decision before this
            // conversion (PromptPrefillChunkAttemptError), shrinking the chunk or
            // raising a typed capacity error. Every other resident path — decode
            // and inject — has no chunk to shrink and no retry to enact, so the
            // decision degrades to the established invalid-request rejection
            // instead of a ResidentForkRequired signal the caller cannot act on.
            AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission { reason, .. } => {
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
        Option<crate::TransientWorkspaceReserveSource>,
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
            // NoLearnedReserve marks request-scoped equations that carry no
            // learned transient evidence; a forward admission cannot produce it,
            // and no source-attribution counter exists for it.
            let admission_reserve_source_counter = match transient_reserve_source {
                crate::TransientWorkspaceReserveSource::ExactContext => {
                    Some(PerformanceCounter::AdmissionReserveExactContextSourceCount)
                }
                crate::TransientWorkspaceReserveSource::PhaseScaled => {
                    Some(PerformanceCounter::AdmissionReservePhaseScaledSourceCount)
                }
                crate::TransientWorkspaceReserveSource::BeyondObservedTokenSpan => {
                    Some(PerformanceCounter::AdmissionReserveBeyondObservedTokenSpanSourceCount)
                }
                crate::TransientWorkspaceReserveSource::GlobalMaximum => {
                    Some(PerformanceCounter::AdmissionReserveGlobalMaximumSourceCount)
                }
                crate::TransientWorkspaceReserveSource::NoLearnedReserve => None,
            };
            if let Some(admission_reserve_source_counter) = admission_reserve_source_counter {
                performance_attribution.record_counter(admission_reserve_source_counter, 1);
            }
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
        let unobserved_scope_transient_floor_bytes = model
            .mlx_ram_budget()
            .adaptive_growth_unobserved_scope_transient_floor_bytes(adaptive_ram_growth_context);
        let forward_projection = self
            .adaptive_ram_growth_guard
            .project_growth_for_context(
                adaptive_ram_growth_context,
                memory_snapshot_before_growth.active_memory_bytes(),
                exact_context_growth_bytes,
                0,
                exact_temporary_workspace_bytes,
                unobserved_scope_transient_floor_bytes,
            )
            .map_err(|projection_error| {
                crate::qwen3_5_resident::model::memory_admission::invalid_request_error(format!(
                    "adaptive RAM growth rejected: {projection_error}"
                ))
            })?;
        let forward_admission_decision = ResidentForwardAdmissionDecision::for_resident_forward(
            &forward_projection,
            matches!(
                adaptive_ram_growth_context.memory_phase(),
                MemoryPhase::Prefill
            )
            .then_some(adaptive_ram_growth_context.forward_token_count()),
        );
        if should_log_memory_decision {
            adaptive_ram_growth_logging::log_adaptive_ram_growth_admission_decision(
                adaptive_ram_growth_context,
                &forward_projection,
                forward_admission_decision.resident_log_label(),
            );
        }
        if let Some(forward_admission_rejection) = forward_admission_decision.rejection() {
            return Err(AdaptiveRamGrowthMemoryAdmissionError::ForwardAdmission {
                reason: resident_forward_admission_reason(
                    &forward_projection,
                    forward_admission_rejection,
                ),
                rejection: forward_admission_rejection,
            });
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

fn resident_forward_admission_reason(
    forward_projection: &AdaptiveRamGrowthProjection,
    forward_admission_rejection: ResidentForwardAdmissionRejection,
) -> String {
    let equation = forward_projection.memory_equation();
    match forward_admission_rejection {
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause: StreamingEngineRequirement::StableShortfall { shortfall_bytes },
        } => format!(
            "resident stable projection of {} bytes exceeds the {} byte ceiling by {shortfall_bytes} bytes",
            equation.stable_projected_bytes(),
            equation.active_memory_ceiling_bytes(),
        ),
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause:
                StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                    shortfall_bytes,
                    prefill_chunk_token_count,
                },
        } => format!(
            "resident prefill chunk of {prefill_chunk_token_count} tokens has a transient peak that exceeds the allowed ceiling by {shortfall_bytes} bytes"
        ),
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause:
                StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                    shortfall_bytes,
                },
        } => format!(
            "resident forward peak of {} bytes exceeds the allowed ceiling of {} bytes by {shortfall_bytes} bytes",
            equation.peak_projected_bytes(),
            equation.allowed_active_memory_bytes(),
        ),
        // The arms below describe causes that never come from
        // `for_resident_forward`: request-context admission rejects before any
        // forward is projected, and runtime allocation failures surface after
        // the forward already ran. The wording states the cause itself because
        // the projection equation the forward never had cannot describe it.
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause: StreamingEngineRequirement::RequestContextShortfall { shortfall_bytes },
        } => format!(
            "request context workspace exceeds the {} byte ceiling by {shortfall_bytes} bytes before any forward ran",
            equation.active_memory_ceiling_bytes(),
        ),
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause: StreamingEngineRequirement::ForwardAllocationShortfall { shortfall_bytes },
        } => format!(
            "runtime forward allocation exceeded the allowed ceiling of {} bytes by {shortfall_bytes} bytes",
            equation.allowed_active_memory_bytes(),
        ),
        ResidentForwardAdmissionRejection::RequireStreamingEngine {
            cause: StreamingEngineRequirement::ForwardGraphicsProcessorMemoryExhausted,
        } => "runtime forward allocation exhausted graphics-processor memory".to_owned(),
    }
}

fn performance_attribution_payload_bytes(performance_attribution: &PerformanceAttribution) -> u64 {
    performance_attribution.expert_streaming_payload_byte_count()
}

#[cfg(test)]
#[path = "memory_admission_tests.rs"]
mod memory_admission_tests;
