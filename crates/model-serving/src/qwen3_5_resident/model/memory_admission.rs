use astronomical_runtime_integration::{MlxMemoryLimits, MlxMemorySnapshot};

use crate::{
    ContextAdmissionRequirements, ExpertWeightMemoryCacheStatistics, InferenceEngineError,
    MemoryAdmissionDecision, StreamingEngineRequirement,
};

use super::super::inference_execution::qwen3_5_runtime_error;
use super::Qwen3_5ResidentModel;

/// Admits context against the current expert owner, then reclaims paged retention.
///
/// Whole-owner demotion is orchestrated by the mutable engine before this model
/// helper runs. Once paged, this helper can reclaim the exact native deficit and
/// retry the unchanged projection before rejecting the user's request.
pub(crate) fn validate_context_memory_admission(
    model: &Qwen3_5ResidentModel,
    memory_limits: MlxMemoryLimits,
    context_memory_reservation_bytes_per_token: usize,
    context_token_count_requiring_reservation: usize,
    temporary_workspace_reservation_bytes: usize,
    additional_maximum_expert_page_reservation_bytes: usize,
) -> Result<(), InferenceEngineError> {
    let initial_projection = build_context_memory_admission_projection(
        model,
        memory_limits,
        context_memory_reservation_bytes_per_token,
        context_token_count_requiring_reservation,
        temporary_workspace_reservation_bytes,
        additional_maximum_expert_page_reservation_bytes,
    )?;
    let context_admission_setup = build_resident_context_admission_setup(
        model,
        initial_projection,
        context_token_count_requiring_reservation,
        temporary_workspace_reservation_bytes,
    );
    crate::memory::log_context_admission_projection(
        "context_admission_after_owner_check",
        context_token_count_requiring_reservation,
        context_memory_reservation_bytes_per_token,
        context_admission_setup.context_admission_requirements,
    );
    let context_admission_decision = context_admission_setup
        .context_admission_requirements
        .decide();
    let context_reclamation_target_bytes = match context_admission_decision {
        MemoryAdmissionDecision::Admit => return Ok(()),
        MemoryAdmissionDecision::Reclaim { required_bytes } => {
            usize::try_from(required_bytes).unwrap_or(usize::MAX)
        }
        MemoryAdmissionDecision::DemoteCompleteResidency { .. } => {
            return Err(resident_context_shortfall_error(
                model,
                context_admission_setup.initial_projected_active_memory_bytes,
                context_admission_setup
                    .reservations
                    .active_memory_ceiling_bytes,
                "generation context requires complete expert demotion before paged admission",
            ));
        }
        MemoryAdmissionDecision::Reject {
            shortfall_bytes, ..
        } => {
            return Err(resident_capacity_error(
                model,
                StreamingEngineRequirement::RequestContextShortfall {
                    shortfall_bytes: usize::try_from(shortfall_bytes).unwrap_or(usize::MAX),
                },
                "generation context exceeds available GPU wired memory",
            ));
        }
    };
    // This flavor parks the post-reclamation snapshot on `None`: whole-owner
    // demotion in the engine is the reclaim path, so a `Reclaim` verdict always
    // falls through to the rejection below.
    if let Some(memory_snapshot_after_reclamation) = None::<MlxMemorySnapshot> {
        let resident_reclaimed_outcome = evaluate_resident_reclaimed_outcome(
            model,
            &context_admission_setup,
            context_reclamation_target_bytes,
            memory_snapshot_after_reclamation,
        );
        return report_resident_reclaimed_outcome(
            model,
            &context_admission_setup,
            resident_reclaimed_outcome,
        );
    }
    Err(reject_resident_context_before_mlx_allocation(
        model,
        &context_admission_setup,
    ))
}

/// Reservation scalars shared by every admission projection in this file.
#[derive(Clone, Copy)]
struct ContextAdmissionReservations {
    context_reservation_bytes: usize,
    maximum_expert_page_reservation_bytes: usize,
    temporary_workspace_reservation_bytes: usize,
    active_memory_ceiling_bytes: usize,
}

/// Resolved inputs the admission decision paths read after the initial projection.
struct ResidentContextAdmissionSetup {
    context_token_count_requiring_reservation: usize,
    initial_memory_snapshot: MlxMemorySnapshot,
    reservations: ContextAdmissionReservations,
    initial_projected_active_memory_bytes: usize,
    expert_weight_memory_cache_statistics_before_reclamation: ExpertWeightMemoryCacheStatistics,
    context_admission_requirements: ContextAdmissionRequirements,
}

fn build_resident_context_admission_setup(
    model: &Qwen3_5ResidentModel,
    initial_projection: ContextMemoryAdmissionProjection,
    context_token_count_requiring_reservation: usize,
    temporary_workspace_reservation_bytes: usize,
) -> ResidentContextAdmissionSetup {
    let initial_memory_snapshot = initial_projection.memory_snapshot;
    let context_reservation_bytes = initial_projection.context_reservation_bytes;
    let maximum_expert_page_reservation_bytes =
        initial_projection.maximum_expert_page_reservation_bytes;
    let configured_mlx_memory_limit_bytes = initial_projection.configured_mlx_memory_limit_bytes;
    let reservations = ContextAdmissionReservations {
        context_reservation_bytes,
        maximum_expert_page_reservation_bytes,
        temporary_workspace_reservation_bytes,
        active_memory_ceiling_bytes: configured_mlx_memory_limit_bytes,
    };
    let initial_projected_active_memory_bytes = ContextAdmissionRequirements {
        current_active_memory_bytes: initial_memory_snapshot.active_memory_bytes(),
        context_growth_bytes: context_reservation_bytes,
        expert_page_reservation_bytes: maximum_expert_page_reservation_bytes,
        temporary_workspace_bytes: temporary_workspace_reservation_bytes,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: configured_mlx_memory_limit_bytes,
        complete_experts_are_resident: model.resident_expert_weights.is_some(),
    }
    .projected_active_memory_bytes()
    .unwrap_or(usize::MAX);
    let expert_weight_memory_cache_statistics_before_reclamation =
        model.expert_weight_memory_cache_statistics();
    let context_admission_requirements = ContextAdmissionRequirements {
        current_active_memory_bytes: initial_memory_snapshot.active_memory_bytes(),
        context_growth_bytes: context_reservation_bytes,
        expert_page_reservation_bytes: maximum_expert_page_reservation_bytes,
        temporary_workspace_bytes: temporary_workspace_reservation_bytes,
        retained_expert_payload_bytes: usize::try_from(
            expert_weight_memory_cache_statistics_before_reclamation.resident_payload_byte_count,
        )
        .unwrap_or(usize::MAX),
        active_memory_ceiling_bytes: configured_mlx_memory_limit_bytes,
        complete_experts_are_resident: model.resident_expert_weights.is_some(),
    };
    ResidentContextAdmissionSetup {
        context_token_count_requiring_reservation,
        initial_memory_snapshot,
        reservations,
        initial_projected_active_memory_bytes,
        expert_weight_memory_cache_statistics_before_reclamation,
        context_admission_requirements,
    }
}

/// Payload bytes actually released by reclamation plus any excess beyond target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ContextReclamationDelta {
    actual_reclaimed_expert_payload_bytes: u64,
    reclamation_overshoot_bytes: u64,
}

fn measure_context_reclamation_delta(
    expert_weight_memory_cache_statistics_before_reclamation: &ExpertWeightMemoryCacheStatistics,
    expert_weight_memory_cache_statistics_after_reclamation: &ExpertWeightMemoryCacheStatistics,
    context_reclamation_target_bytes: usize,
) -> ContextReclamationDelta {
    let actual_reclaimed_expert_payload_bytes =
        expert_weight_memory_cache_statistics_before_reclamation
            .resident_payload_byte_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
            );
    let reclamation_overshoot_bytes = actual_reclaimed_expert_payload_bytes
        .saturating_sub(u64::try_from(context_reclamation_target_bytes).unwrap_or(u64::MAX));
    ContextReclamationDelta {
        actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes,
    }
}

/// Post-reclamation verdict for the unchanged context projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReclaimedContextAdmissionProjection {
    projected_active_memory_bytes_after_reclamation: usize,
    post_reclamation_decision: MemoryAdmissionDecision,
}

fn project_reclaimed_context_admission(
    reservations: ContextAdmissionReservations,
    active_memory_bytes_after_reclamation: usize,
    retained_expert_payload_bytes_after_reclamation: usize,
) -> ReclaimedContextAdmissionProjection {
    // Complete residency is already demoted by the time a reclaim target
    // exists, so the re-projection never charges a complete-residency owner.
    let post_reclamation_requirements = ContextAdmissionRequirements {
        current_active_memory_bytes: active_memory_bytes_after_reclamation,
        context_growth_bytes: reservations.context_reservation_bytes,
        expert_page_reservation_bytes: reservations.maximum_expert_page_reservation_bytes,
        temporary_workspace_bytes: reservations.temporary_workspace_reservation_bytes,
        retained_expert_payload_bytes: retained_expert_payload_bytes_after_reclamation,
        active_memory_ceiling_bytes: reservations.active_memory_ceiling_bytes,
        complete_experts_are_resident: false,
    };
    let projected_active_memory_bytes_after_reclamation = post_reclamation_requirements
        .projected_active_memory_bytes()
        .unwrap_or(usize::MAX);
    let post_reclamation_decision = post_reclamation_requirements.decide();
    ReclaimedContextAdmissionProjection {
        projected_active_memory_bytes_after_reclamation,
        post_reclamation_decision,
    }
}

/// Post-reclamation state one reporting pass consumes.
struct ResidentReclaimedOutcome {
    memory_snapshot_after_reclamation: MlxMemorySnapshot,
    expert_weight_memory_cache_statistics_after_reclamation: ExpertWeightMemoryCacheStatistics,
    actual_reclaimed_expert_payload_bytes: u64,
    reclamation_overshoot_bytes: u64,
    projected_active_memory_bytes_after_reclamation: usize,
    post_reclamation_decision: MemoryAdmissionDecision,
}

fn evaluate_resident_reclaimed_outcome(
    model: &Qwen3_5ResidentModel,
    context_admission_setup: &ResidentContextAdmissionSetup,
    context_reclamation_target_bytes: usize,
    memory_snapshot_after_reclamation: MlxMemorySnapshot,
) -> ResidentReclaimedOutcome {
    let expert_weight_memory_cache_statistics_after_reclamation =
        model.expert_weight_memory_cache_statistics();
    let context_reclamation_delta = measure_context_reclamation_delta(
        &context_admission_setup.expert_weight_memory_cache_statistics_before_reclamation,
        &expert_weight_memory_cache_statistics_after_reclamation,
        context_reclamation_target_bytes,
    );
    let post_reclamation_projection = project_reclaimed_context_admission(
        context_admission_setup.reservations,
        memory_snapshot_after_reclamation.active_memory_bytes(),
        usize::try_from(
            expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
        )
        .unwrap_or(usize::MAX),
    );
    ResidentReclaimedOutcome {
        memory_snapshot_after_reclamation,
        expert_weight_memory_cache_statistics_after_reclamation,
        actual_reclaimed_expert_payload_bytes: context_reclamation_delta
            .actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes: context_reclamation_delta.reclamation_overshoot_bytes,
        projected_active_memory_bytes_after_reclamation: post_reclamation_projection
            .projected_active_memory_bytes_after_reclamation,
        post_reclamation_decision: post_reclamation_projection.post_reclamation_decision,
    }
}

fn report_resident_reclaimed_outcome(
    model: &Qwen3_5ResidentModel,
    context_admission_setup: &ResidentContextAdmissionSetup,
    resident_reclaimed_outcome: ResidentReclaimedOutcome,
) -> Result<(), InferenceEngineError> {
    let context_token_count_requiring_reservation =
        context_admission_setup.context_token_count_requiring_reservation;
    let initial_memory_snapshot = context_admission_setup.initial_memory_snapshot;
    let context_reservation_bytes = context_admission_setup
        .reservations
        .context_reservation_bytes;
    let maximum_expert_page_reservation_bytes = context_admission_setup
        .reservations
        .maximum_expert_page_reservation_bytes;
    let temporary_workspace_reservation_bytes = context_admission_setup
        .reservations
        .temporary_workspace_reservation_bytes;
    let configured_mlx_memory_limit_bytes = context_admission_setup
        .reservations
        .active_memory_ceiling_bytes;
    let expert_weight_memory_cache_statistics_before_reclamation =
        context_admission_setup.expert_weight_memory_cache_statistics_before_reclamation;
    let ResidentReclaimedOutcome {
        memory_snapshot_after_reclamation,
        expert_weight_memory_cache_statistics_after_reclamation,
        actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes,
        projected_active_memory_bytes_after_reclamation,
        post_reclamation_decision,
    } = resident_reclaimed_outcome;
    if post_reclamation_decision == MemoryAdmissionDecision::Admit {
        tracing::info!(
            context_token_count_requiring_reservation,
            initial_active_memory_bytes = initial_memory_snapshot.active_memory_bytes(),
            active_memory_bytes_after_reclamation =
                memory_snapshot_after_reclamation.active_memory_bytes(),
            context_reservation_bytes,
            maximum_expert_page_reservation_bytes,
            temporary_workspace_reservation_bytes,
            projected_active_memory_bytes_after_reclamation,
            configured_mlx_memory_limit_bytes,
            retained_expert_payload_bytes_before =
                expert_weight_memory_cache_statistics_before_reclamation
                    .resident_payload_byte_count,
            retained_expert_payload_bytes_after =
                expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
            actual_reclaimed_expert_payload_bytes,
            reclamation_overshoot_bytes,
            expert_eviction_count_delta = expert_weight_memory_cache_statistics_after_reclamation
                .eviction_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_before_reclamation.eviction_count,
                ),
            allocator_cache_memory_bytes_after_reclamation =
                memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
            "admitted generation context after reclaiming retained experts"
        );
        return Ok(());
    }
    tracing::warn!(
        context_token_count_requiring_reservation,
        initial_active_memory_bytes = initial_memory_snapshot.active_memory_bytes(),
        active_memory_bytes_after_reclamation =
            memory_snapshot_after_reclamation.active_memory_bytes(),
        context_reservation_bytes,
        maximum_expert_page_reservation_bytes,
        temporary_workspace_reservation_bytes,
        projected_active_memory_bytes_after_reclamation,
        configured_mlx_memory_limit_bytes,
        retained_expert_payload_bytes_before =
            expert_weight_memory_cache_statistics_before_reclamation.resident_payload_byte_count,
        retained_expert_payload_bytes_after =
            expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
        actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes,
        expert_eviction_count_delta = expert_weight_memory_cache_statistics_after_reclamation
            .eviction_count
            .saturating_sub(
                expert_weight_memory_cache_statistics_before_reclamation.eviction_count,
            ),
        allocator_cache_memory_bytes_after_reclamation =
            memory_snapshot_after_reclamation.allocator_cache_memory_bytes(),
        "rejected generation context after retained-expert reclamation remained insufficient"
    );
    Err(resident_capacity_error(
        model,
        StreamingEngineRequirement::RequestContextShortfall {
            shortfall_bytes: projected_active_memory_bytes_after_reclamation
                .saturating_sub(configured_mlx_memory_limit_bytes),
        },
        "generation context exceeds available GPU wired memory",
    ))
}

fn reject_resident_context_before_mlx_allocation(
    model: &Qwen3_5ResidentModel,
    context_admission_setup: &ResidentContextAdmissionSetup,
) -> InferenceEngineError {
    let context_token_count_requiring_reservation =
        context_admission_setup.context_token_count_requiring_reservation;
    let initial_memory_snapshot = context_admission_setup.initial_memory_snapshot;
    let context_reservation_bytes = context_admission_setup
        .reservations
        .context_reservation_bytes;
    let maximum_expert_page_reservation_bytes = context_admission_setup
        .reservations
        .maximum_expert_page_reservation_bytes;
    let temporary_workspace_reservation_bytes = context_admission_setup
        .reservations
        .temporary_workspace_reservation_bytes;
    let initial_projected_active_memory_bytes =
        context_admission_setup.initial_projected_active_memory_bytes;
    let configured_mlx_memory_limit_bytes = context_admission_setup
        .reservations
        .active_memory_ceiling_bytes;
    tracing::warn!(
        context_token_count_requiring_reservation,
        live_memory_bytes = initial_memory_snapshot.active_memory_bytes(),
        context_reservation_bytes,
        maximum_expert_page_reservation_bytes,
        temporary_workspace_reservation_bytes,
        projected_active_memory_bytes = initial_projected_active_memory_bytes,
        configured_mlx_memory_limit_bytes,
        "rejected generation context before MLX request allocation"
    );
    resident_context_shortfall_error(
        model,
        initial_projected_active_memory_bytes,
        configured_mlx_memory_limit_bytes,
        "generation context exceeds available GPU wired memory",
    )
}

fn resident_context_shortfall_error(
    model: &Qwen3_5ResidentModel,
    initial_projected_active_memory_bytes: usize,
    configured_mlx_memory_limit_bytes: usize,
    reason: &'static str,
) -> InferenceEngineError {
    resident_capacity_error(
        model,
        StreamingEngineRequirement::RequestContextShortfall {
            shortfall_bytes: initial_projected_active_memory_bytes
                .saturating_sub(configured_mlx_memory_limit_bytes),
        },
        reason,
    )
}

struct ContextMemoryAdmissionProjection {
    memory_snapshot: MlxMemorySnapshot,
    context_reservation_bytes: usize,
    maximum_expert_page_reservation_bytes: usize,
    configured_mlx_memory_limit_bytes: usize,
}

fn build_context_memory_admission_projection(
    model: &Qwen3_5ResidentModel,
    memory_limits: MlxMemoryLimits,
    context_memory_reservation_bytes_per_token: usize,
    context_token_count_requiring_reservation: usize,
    _temporary_workspace_reservation_bytes: usize,
    additional_maximum_expert_page_reservation_bytes: usize,
) -> Result<ContextMemoryAdmissionProjection, InferenceEngineError> {
    let memory_snapshot = model
        .runtime()
        .memory_snapshot()
        .map_err(qwen3_5_runtime_error)?;
    let context_reservation_bytes = context_token_count_requiring_reservation
        .checked_mul(context_memory_reservation_bytes_per_token)
        .ok_or_else(|| invalid_request_error("generation context memory reservation overflowed"))?;
    // Resident forwards need no route reserve because all expert arrays already
    // contribute to active memory. Paged forwards reserve one largest top-K page.
    let maximum_target_expert_page_reservation_bytes = 0;
    let maximum_expert_page_reservation_bytes =
        usize::try_from(maximum_target_expert_page_reservation_bytes)
            .map_err(|_| {
                invalid_request_error("maximum expert page reservation exceeds the platform range")
            })?
            .checked_add(additional_maximum_expert_page_reservation_bytes)
            .ok_or_else(|| invalid_request_error("combined expert page reservation overflowed"))?;
    Ok(ContextMemoryAdmissionProjection {
        memory_snapshot,
        context_reservation_bytes,
        maximum_expert_page_reservation_bytes,
        configured_mlx_memory_limit_bytes: memory_limits.active_memory_limit_bytes(),
    })
}

pub(crate) fn invalid_request_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.into(),
    }
}

pub(in crate::qwen3_5_resident) fn resident_capacity_error(
    model: &Qwen3_5ResidentModel,
    cause: StreamingEngineRequirement,
    reason: impl Into<String>,
) -> InferenceEngineError {
    let reason = reason.into();
    if model.resident_expert_weights.is_some() {
        InferenceEngineError::ResidentForkRequired { cause, reason }
    } else {
        InferenceEngineError::InvalidRequest { reason }
    }
}

#[cfg(test)]
#[path = "memory_admission_tests.rs"]
mod memory_admission_tests;
