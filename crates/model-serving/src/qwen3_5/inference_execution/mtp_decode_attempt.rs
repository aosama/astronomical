use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::multi_token_prediction::{
    MtpDraftDepth, attempt_prediction_proposal_and_verification,
    disable_prediction_after_memory_admission_failure, effective_prediction_depth,
    projected_verification_window_memory_growth_bytes, verification_boundary_snapshot_bytes,
    verification_transient_array_bytes,
};
use crate::{AdaptiveRamGrowthContext, InferenceEngineError, PerformanceCounter};

use super::advance_generation::ActiveRequestAdvance;
use super::completed_forward_memory::{
    collect_completed_forward_memory_snapshot, record_completed_adaptive_ram_growth,
};
use super::engine_request::Qwen3_5EngineRequest;
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::{Qwen3_5EngineState, fatal_engine_error};

impl Qwen3_5EngineState {
    pub(super) fn attempt_mtp_decode_window(
        &mut self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        current_generated_token: &MlxArray,
        current_generated_token_id: u32,
    ) -> Result<Option<ActiveRequestAdvance>, InferenceEngineError> {
        // Session creation already refuses SSD-paged sparse experts. Keep this
        // decode-time gate so a later demotion cannot open a two-row verify
        // against an incomplete expert set.
        if self
            .model
            .as_ref()
            .is_some_and(|model| model.sparse_experts_are_paged())
        {
            return Ok(None);
        }
        let (window_clamped_depth, window_downgrade_reason) =
            effective_prediction_depth(active_request, self.maximum_position_count);
        if let Some(window_downgrade_reason) = window_downgrade_reason {
            let downgrade_counter = match window_downgrade_reason {
                crate::qwen3_5::MtpDepthDowngradeReason::OutputWindow => {
                    PerformanceCounter::MtpOutputDepthDowngradeCount
                }
                crate::qwen3_5::MtpDepthDowngradeReason::ContextWindow => {
                    PerformanceCounter::MtpContextDepthDowngradeCount
                }
                crate::qwen3_5::MtpDepthDowngradeReason::ThinkingWindow => {
                    PerformanceCounter::MtpThinkingDepthDowngradeCount
                }
                crate::qwen3_5::MtpDepthDowngradeReason::Memory => {
                    PerformanceCounter::MtpMemoryDepthDowngradeCount
                }
            };
            // Window clamping occurs before graph construction, so this counter attributes the
            // avoided deeper attempt without charging an MTP operational fallback.
            active_request
                .performance_attribution_mut()
                .record_counter(downgrade_counter, 1);
        }
        let Some(window_clamped_depth) = window_clamped_depth else {
            return Ok(None);
        };
        let mut candidate_depth_value = window_clamped_depth.get();
        let mut last_rejected_memory_projection = None;
        while candidate_depth_value >= MtpDraftDepth::MINIMUM {
            let candidate_depth = MtpDraftDepth::new(candidate_depth_value)
                .map_err(|_| fatal_engine_error("candidate MTP depth is outside 1 through 3"))?;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let prediction_history_growth_bytes =
                projected_verification_window_memory_growth_bytes(
                    model,
                    active_request,
                    candidate_depth,
                )?;
            let boundary_snapshot_bytes =
                verification_boundary_snapshot_bytes(model, candidate_depth)?;
            let verification_transient_array_bytes =
                verification_transient_array_bytes(model, candidate_depth)?;
            // Admission charges snapshots and transient arrays together because they coexist.
            // Attribution records them on separate counters so one owner cannot hide the other.
            let verification_workspace_bytes = boundary_snapshot_bytes
                .checked_add(verification_transient_array_bytes)
                .ok_or_else(|| fatal_engine_error("MTP verification workspace overflowed"))?;
            let growth_context = AdaptiveRamGrowthContext::decode(
                usize::from(candidate_depth.get()) + 1,
                true,
                model.sparse_experts_are_paged(),
            );
            let memory_admission = self.measure_adaptive_ram_growth_memory_admission(
                growth_context,
                &mut active_request.performance_attribution,
                &active_request.request_decoder_state,
                prediction_history_growth_bytes,
                verification_workspace_bytes,
            );
            let (active_memory_bytes_before_growth, retained_expert_payload_bytes_before_growth) =
                match memory_admission {
                    Ok(admitted_memory) => admitted_memory,
                    Err(AdaptiveRamGrowthMemoryAdmissionError::Engine(memory_error)) => {
                        return Err(memory_error);
                    }
                    Err(AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason }) => {
                        last_rejected_memory_projection = Some((
                            prediction_history_growth_bytes,
                            verification_transient_array_bytes,
                            boundary_snapshot_bytes,
                        ));
                        // Rejected depths keep one downgrade count each. Byte projections wait
                        // for the admitted depth or the final target-only fallback below.
                        active_request
                            .performance_attribution_mut()
                            .record_counter(PerformanceCounter::MtpMemoryDepthDowngradeCount, 1);
                        tracing::info!(
                            request_id = request_id.value(),
                            candidate_depth = candidate_depth.get(),
                            reason,
                            "MTP depth did not fit memory admission; trying a shallower depth"
                        );
                        candidate_depth_value = candidate_depth_value.saturating_sub(1);
                        continue;
                    }
                };
            record_mtp_memory_projection(
                active_request,
                prediction_history_growth_bytes,
                verification_transient_array_bytes,
                boundary_snapshot_bytes,
            );
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
            let decision = attempt_prediction_proposal_and_verification(
                model,
                active_request,
                request_id,
                current_generated_token,
                current_generated_token_id,
                candidate_depth,
                &self.end_of_sequence_token_ids,
            )?;
            if decision.is_operational_fallback() {
                record_completed_adaptive_ram_growth(
                    &mut self.adaptive_ram_growth_guard,
                    growth_context.with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                    true,
                    model,
                    active_memory_bytes_before_growth,
                    retained_expert_payload_bytes_before_growth,
                    verification_workspace_bytes,
                    &mut active_request.performance_attribution,
                )?;
                return Ok(None);
            }
            let memory_observation = collect_completed_forward_memory_snapshot(
                &mut self.adaptive_ram_growth_guard,
                growth_context.with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
                true,
                model,
                active_memory_bytes_before_growth,
                retained_expert_payload_bytes_before_growth,
                verification_workspace_bytes,
                &mut active_request.performance_attribution,
            )?;
            let emission = self.build_generated_token_emission(
                model,
                active_request,
                current_generated_token_id,
                Some(&memory_observation),
            )?;
            return Ok(Some(if emission.is_terminal {
                ActiveRequestAdvance::Complete(emission.generated_token)
            } else {
                ActiveRequestAdvance::Continue(emission.generated_token)
            }));
        }
        if let Some((
            prediction_history_growth_bytes,
            verification_transient_array_bytes,
            boundary_snapshot_bytes,
        )) = last_rejected_memory_projection
        {
            record_mtp_memory_projection(
                active_request,
                prediction_history_growth_bytes,
                verification_transient_array_bytes,
                boundary_snapshot_bytes,
            );
        }
        disable_prediction_after_memory_admission_failure(active_request);
        Ok(None)
    }
}

fn record_mtp_memory_projection(
    active_request: &mut Qwen3_5EngineRequest,
    persistent_growth_bytes: usize,
    verification_workspace_bytes: usize,
    boundary_snapshot_bytes: usize,
) {
    for (counter, byte_count) in [
        (
            PerformanceCounter::MtpPersistentGrowthByteCount,
            persistent_growth_bytes,
        ),
        (
            PerformanceCounter::MtpVerificationWorkspaceByteCount,
            verification_workspace_bytes,
        ),
        (
            PerformanceCounter::MtpBoundarySnapshotByteCount,
            boundary_snapshot_bytes,
        ),
    ] {
        active_request
            .performance_attribution_mut()
            .record_counter(counter, u64::try_from(byte_count).unwrap_or(u64::MAX));
    }
}
