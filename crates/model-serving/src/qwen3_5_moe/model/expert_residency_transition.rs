//! Whole-model transitions between contiguous residency and native paging.
//!
//! # Two ownership modes a later reader must not mix
//!
//! - Complete residency: every mixture-of-experts weight sits in RAM as one
//!   owner (`resident_expert_weights = Some`). Decode never reads experts from
//!   disk. This owner is atomic: paging cannot evict three specialists and keep
//!   the rest.
//! - Paged mode: expert weights live on disk. A layer either streams for one
//!   operation or keeps a smaller demand-selected page in
//!   `retained_experts`.
//!
//! # Safe orders
//!
//! Promotion follows prepare -> build candidate -> publish. Native retention is
//! frozen before its pages are reclaimed, and the model remains observably paged
//! until a complete candidate exists. Demotion follows synchronize -> unpublish
//! -> drop -> clear allocator -> resume paging. These orders prevent lazy MLX
//! work from retaining released arrays and make every failure state usable.
//!
//! # Replacement-aware admission
//!
//! `projected active = current active - retained paged experts + complete experts`
//!
//! Current active memory already owns the hot paged experts. Adding the complete
//! payload without subtracting those pages would count the same expert category
//! twice and reject a valid promotion. Reclaiming the pages before evaluating the
//! formula has the opposite failure: an impossible promotion destroys the hot set
//! and makes the next request read the same experts from disk again. Therefore the
//! fit decision must happen first, while the retained pages still have an owner.
//!
//! There is no sticky "stay paged" flag. Startup may install the atomic complete
//! owner when it fits; request pressure may demote it. During a request, phase-aware
//! read-through retention preserves complete and routed layer pages. After request
//! workspace is released, complete residency is restored when the leftover ceiling admits it.

use astronomical_runtime_integration::MlxRuntimeError;

use crate::qwen3_5::model::{Qwen3_5ExecutionError, Qwen3_5Model};
use crate::qwen3_5_moe::Qwen3_5ResidentExpertWeights;
use crate::qwen3_5_moe::expert_residency::maximum_resident_gate_up_fusion_transient_payload_bytes;
use crate::{
    CompleteResidencyDecision, CompleteResidencyRequirements, PerformanceAttribution,
    PerformanceOperation, required_complete_residency_activation_headroom_bytes,
};

/// Why the owner thread asked for a whole-model expert residency change.
///
/// These labels exist for logs and attribution, not for sticky policy. The
/// leftover ceiling still decides whether promotion fits. Do not add a variant
/// that means "never promote again".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5ExpertResidencyTransitionReason {
    /// First load: try complete residency before the first user request.
    Startup,
    /// A new request projected that complete experts still fit beside context.
    RequestAdmission,
    /// The user raised the public memory ceiling while the model was idle.
    CeilingRaise,
    /// Prefill activations did not fit, so demote the complete owner now.
    RequestPressure,
    /// The user lowered the public memory ceiling.
    CeilingLower,
    /// Draft-model loading needs the target to yield or reclaim expert RAM.
    SpeculativePrefillDraftLoading,
    /// The request released its workspace; restore complete residency if it fits.
    RequestCompletion,
    /// Prefill finished; restore complete residency if generate still fits.
    DecodeHandoff,
}

/// Nonfatal outcomes from an optional complete-model promotion attempt.
///
/// A `DoesNotFit` or `RecoverableCapacityRejection` result is a normal paged
/// outcome. The caller must keep serving the user from disk or demand-selected
/// pages. Only a structural `Err` is a real failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5ExpertResidencyPromotionOutcome {
    /// Dense models, or sparse models that already own every expert.
    AlreadyResident,
    /// The complete owner is now published and paged pages were released.
    Promoted,
    /// The leftover ceiling plus required activation headroom is too small.
    DoesNotFit,
    /// Native allocation rejected the candidate after admission. Paging remains.
    RecoverableCapacityRejection,
}

struct CompleteExpertResidencyHeadroom {
    observed_transient_high_water_bytes: u64,
    required_activation_headroom_bytes: u64,
    gate_up_fusion_transient_payload_bytes: u64,
    required_residency_headroom_bytes: u64,
}

impl Qwen3_5Model {
    pub(crate) fn required_complete_expert_residency_headroom_bytes(
        &self,
    ) -> Result<u64, Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(0);
        };
        let complete_expert_payload_bytes = expert_pager.complete_expert_payload_byte_count()?;
        Ok(self
            .complete_expert_residency_headroom(complete_expert_payload_bytes)?
            .required_residency_headroom_bytes)
    }

    fn complete_expert_residency_headroom(
        &self,
        _complete_expert_payload_bytes: u64,
    ) -> Result<CompleteExpertResidencyHeadroom, Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(CompleteExpertResidencyHeadroom {
                observed_transient_high_water_bytes: 0,
                required_activation_headroom_bytes: 0,
                gate_up_fusion_transient_payload_bytes: 0,
                required_residency_headroom_bytes: 0,
            });
        };
        let observed_transient_high_water_bytes =
            expert_pager.observed_transient_high_water_bytes();
        let gate_up_fusion_transient_payload_bytes =
            maximum_resident_gate_up_fusion_transient_payload_bytes(expert_pager.layer_plans())?;
        // Promotion replaces paged pages with complete weights. Prefill's
        // three-layer weight heuristic is not extra RAM on top of that payload;
        // only fusion transients and measured decode working set coexist with it.
        let required_activation_headroom_bytes =
            required_complete_residency_activation_headroom_bytes(
                gate_up_fusion_transient_payload_bytes,
                observed_transient_high_water_bytes,
            );
        Ok(CompleteExpertResidencyHeadroom {
            observed_transient_high_water_bytes,
            required_activation_headroom_bytes,
            gate_up_fusion_transient_payload_bytes,
            required_residency_headroom_bytes: required_activation_headroom_bytes,
        })
    }

    pub(crate) fn try_promote_experts_to_resident(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExecutionError> {
        performance_attribution.measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |performance_attribution| {
                self.try_promote_experts_to_resident_without_attribution(
                    transition_reason,
                    performance_attribution.positional_file_read_metrics(),
                )
            },
        )
    }

    fn try_promote_experts_to_resident_without_attribution(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        positional_file_read_metrics: Option<
            std::sync::Arc<astronomical_runtime_integration::PositionalFileReadMetrics>,
        >,
    ) -> Result<Qwen3_5ExpertResidencyPromotionOutcome, Qwen3_5ExecutionError> {
        let Some(expert_pager) = self.expert_pager.as_ref() else {
            return Ok(Qwen3_5ExpertResidencyPromotionOutcome::AlreadyResident);
        };
        if self.resident_expert_weights.is_some() {
            return Ok(Qwen3_5ExpertResidencyPromotionOutcome::AlreadyResident);
        }
        // Decode handoff and request finalization may restore the complete
        // owner when the leftover ceiling admits it. Fit is decided below, not
        // by a sticky paged flag from the earlier demotion.
        let complete_expert_payload_bytes = expert_pager.complete_expert_payload_byte_count()?;
        tracing::info!(
            ?transition_reason,
            complete_expert_payload_bytes,
            "started complete-model expert residency admission"
        );

        // Finish submitted products before sampling idle memory or replacing the
        // complete resident expert owner.
        self.runtime.synchronize_gpu_stream()?;

        // Everything inside this closure is speculative. `self` is not changed
        // until the complete candidate reaches the publication point below. In
        // particular, `resident_expert_weights` remains None, so any early return
        // leaves the model truthfully and safely in paged mode.
        let candidate_resident_expert_weights_result = (|| {
            // Idle promotion replaces any complete layers retained by paged mode.
            let retained_streamed_expert_payload_bytes =
                self.retained_experts
                    .as_ref()
                    .map_or(0, |retained_experts| {
                        retained_experts
                            .borrow()
                            .statistics()
                            .resident_payload_byte_count
                    });

            // Sample after stream synchronization. Sampling
            // earlier could include unfinished request work or race a later page
            // insertion, making the replacement projection internally inconsistent.
            let idle_memory_snapshot = self.runtime.memory_snapshot()?;
            let idle_active_memory_bytes =
                u64::try_from(idle_memory_snapshot.active_memory_bytes()).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "idle MLX active memory exceeds the u64 range",
                    }
                })?;
            let stable_memory_ceiling_bytes = self
                .mlx_ram_budget
                .borrow()
                .mlx_active_memory_ceiling_bytes();

            // Static payload fit is not enough. Prefill activations, key-value
            // growth, and temporary workspaces still need spare ceiling after
            // complete experts occupy memory. Without this reservation the model
            // promotes into a ceiling it cannot serve from and thrashes near the
            // MLX active-memory limit.
            // Resident fusion and request activations occur in disjoint phases,
            // so the shared policy helper reserves whichever phase has the larger peak.
            let residency_headroom =
                self.complete_expert_residency_headroom(complete_expert_payload_bytes)?;
            let required_residency_headroom_bytes =
                residency_headroom.required_residency_headroom_bytes;
            let complete_residency_decision = CompleteResidencyRequirements {
                current_active_memory_bytes: idle_active_memory_bytes,
                retained_paged_expert_payload_bytes: retained_streamed_expert_payload_bytes,
                complete_expert_payload_bytes,
                required_headroom_bytes: required_residency_headroom_bytes,
                active_memory_ceiling_bytes: stable_memory_ceiling_bytes,
            }
            .decide();
            match complete_residency_decision {
                CompleteResidencyDecision::Admit {
                    projected_active_memory_bytes,
                    ..
                } => {
                    tracing::debug!(
                        ?transition_reason,
                        projected_active_memory_bytes,
                        required_activation_headroom_bytes =
                            residency_headroom.required_activation_headroom_bytes,
                        gate_up_fusion_transient_payload_bytes =
                            residency_headroom.gate_up_fusion_transient_payload_bytes,
                        required_residency_headroom_bytes,
                        stable_memory_ceiling_bytes,
                        "admitted complete-model expert residency replacement"
                    );
                }
                CompleteResidencyDecision::DoesNotFit {
                    projected_active_memory_bytes,
                    shortfall_bytes,
                    boundary,
                    ..
                } => {
                    tracing::info!(
                        ?transition_reason,
                        idle_active_memory_bytes,
                        retained_streamed_expert_payload_bytes,
                        complete_expert_payload_bytes,
                        projected_resident_active_memory_bytes = projected_active_memory_bytes,
                        observed_transient_high_water_bytes =
                            residency_headroom.observed_transient_high_water_bytes,
                        required_activation_headroom_bytes =
                            residency_headroom.required_activation_headroom_bytes,
                        gate_up_fusion_transient_payload_bytes =
                            residency_headroom.gate_up_fusion_transient_payload_bytes,
                        required_residency_headroom_bytes,
                        stable_memory_ceiling_bytes,
                        ?boundary,
                        shortfall_bytes,
                        outcome = "does_not_fit",
                        "completed complete-model expert residency admission"
                    );
                    return Ok(None);
                }
                CompleteResidencyDecision::RejectInvalidObservation { error } => {
                    tracing::warn!(
                        ?transition_reason,
                        %error,
                        outcome = "invalid_memory_observation",
                        "rejected complete-model expert residency admission"
                    );
                    return Err(Qwen3_5ExecutionError::InvalidInput {
                        description: "complete-residency memory observation was inconsistent",
                    });
                }
            }
            // This point is reached only after the memory package admitted the
            // replacement. Execution below performs ownership mutation and I/O.

            // Crossing this point commits to replacing the paged representation.
            // Only now may policy remove the hot pages, because exact accounting
            // proved that their complete replacement fits under the same ceiling.
            self.runtime
                .synchronize_gpu_stream_and_clear_allocator_cache()?;

            if let Some(retained_experts) = self.retained_experts.as_ref() {
                retained_experts.borrow_mut().release_all();
            }

            // Build every resident layer into a candidate owner. Publication is
            // still delayed until the match below confirms the complete load.
            Qwen3_5ResidentExpertWeights::load(self, positional_file_read_metrics).map(Some)
        })();
        let candidate_resident_expert_weights = match candidate_resident_expert_weights_result {
            Ok(Some(candidate_resident_expert_weights)) => candidate_resident_expert_weights,
            Ok(None) => {
                return Ok(Qwen3_5ExpertResidencyPromotionOutcome::DoesNotFit);
            }
            Err(resident_loading_error) => {
                // A failure after replacement began may leave candidate buffers
                // in MLX allocator storage. Clear them and always restore paging
                // growth so the fallback model remains usable.
                let is_recoverable_capacity_rejection =
                    resident_loading_error_is_recoverable_capacity(&resident_loading_error);
                let cleanup_result = self
                    .runtime
                    .synchronize_gpu_stream_and_clear_allocator_cache();
                if let Err(cleanup_error) = cleanup_result {
                    return Err(cleanup_error.into());
                }
                if is_recoverable_capacity_rejection {
                    tracing::info!(
                        ?transition_reason,
                        complete_expert_payload_bytes,
                        outcome = "recoverable_capacity_rejection",
                        error = %resident_loading_error,
                        "completed complete-model expert residency admission"
                    );
                    return Ok(
                        Qwen3_5ExpertResidencyPromotionOutcome::RecoverableCapacityRejection,
                    );
                }
                return Err(resident_loading_error);
            }
        };
        // This assignment is the only Paged -> Resident publication point. No
        // observer can report Resident while only some layers have loaded.
        self.resident_expert_weights = Some(candidate_resident_expert_weights);
        tracing::info!(
            ?transition_reason,
            complete_expert_payload_bytes,
            resident_layer_count = self
                .resident_expert_weights
                .as_ref()
                .map_or(0, Qwen3_5ResidentExpertWeights::layer_count),
            outcome = "promoted",
            "completed complete-model expert residency admission"
        );
        Ok(Qwen3_5ExpertResidencyPromotionOutcome::Promoted)
    }

    pub(crate) fn demote_resident_experts_to_paging(
        &mut self,
        transition_reason: Qwen3_5ExpertResidencyTransitionReason,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<bool, Qwen3_5ExecutionError> {
        if self.resident_expert_weights.is_none() {
            return Ok(false);
        }
        performance_attribution.measure_operation(
            PerformanceOperation::MlxAllocatorCacheCleanup,
            |_performance_attribution| {
                // Synchronize before removing the owner because lazy arrays and
                // submitted kernels may still reference its backing allocations.
                self.runtime.synchronize_gpu_stream()?;
                let released_resident_expert_weights = self.resident_expert_weights.take();
                let released_resident_expert_payload_bytes = released_resident_expert_weights
                    .as_ref()
                    .map_or(0, Qwen3_5ResidentExpertWeights::payload_byte_count);
                drop(released_resident_expert_weights);
                // Dropped resident buffers become reusable allocator storage;
                // clearing it makes the newly paged mode's capacity observable.
                let allocator_cleanup_result = self.runtime.clear_allocator_cache();
                allocator_cleanup_result?;
                tracing::info!(
                    ?transition_reason,
                    released_resident_expert_payload_bytes,
                    "demoted complete resident experts to Rust streaming"
                );
                Ok(true)
            },
        )
    }
}

fn resident_loading_error_is_recoverable_capacity(
    resident_loading_error: &Qwen3_5ExecutionError,
) -> bool {
    match resident_loading_error {
        Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded { .. }) => true,
        Qwen3_5ExecutionError::Runtime(runtime_error) => {
            runtime_error.is_recoverable_graphics_processor_out_of_memory()
        }
        _ => false,
    }
}
