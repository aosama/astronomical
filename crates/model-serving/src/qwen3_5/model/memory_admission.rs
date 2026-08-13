use astronomical_runtime_integration::{MlxMemoryLimits, MlxMemorySnapshot};

use crate::{ContextAdmissionRequirements, InferenceEngineError, MemoryAdmissionDecision};

use super::super::inference_execution::qwen3_5_runtime_error;
use super::Qwen3_5Model;
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;

/// Admits context against the current expert owner, then reclaims paged retention.
///
/// Whole-owner demotion is orchestrated by the mutable engine before this model
/// helper runs. Once paged, this helper can reclaim the exact native deficit and
/// retry the unchanged projection before rejecting the user's request.
pub(crate) fn validate_context_memory_admission(
    model: &Qwen3_5Model,
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
    let initial_memory_snapshot = initial_projection.memory_snapshot;
    let context_reservation_bytes = initial_projection.context_reservation_bytes;
    let maximum_expert_page_reservation_bytes =
        initial_projection.maximum_expert_page_reservation_bytes;
    let configured_mlx_memory_limit_bytes = initial_projection.configured_mlx_memory_limit_bytes;
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
    let context_admission_decision = ContextAdmissionRequirements {
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
    }
    .decide();
    let context_reclamation_target_bytes = match context_admission_decision {
        MemoryAdmissionDecision::Admit => return Ok(()),
        MemoryAdmissionDecision::Reclaim { required_bytes } => {
            usize::try_from(required_bytes).unwrap_or(usize::MAX)
        }
        MemoryAdmissionDecision::DemoteCompleteResidency { .. } => {
            return Err(invalid_request_error(
                "generation context requires complete expert demotion before paged admission",
            ));
        }
        MemoryAdmissionDecision::Reject { .. } => {
            return Err(invalid_request_error(
                "generation context exceeds available GPU wired memory",
            ));
        }
    };
    if let Some(memory_snapshot_after_reclamation) =
        reclaim_retained_experts_for_request_memory_pressure(
            model,
            context_reclamation_target_bytes,
        )?
    {
        let expert_weight_memory_cache_statistics_after_reclamation =
            model.expert_weight_memory_cache_statistics();
        let actual_reclaimed_expert_payload_bytes =
            expert_weight_memory_cache_statistics_before_reclamation
                .resident_payload_byte_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_after_reclamation
                        .resident_payload_byte_count,
                );
        let reclamation_overshoot_bytes = actual_reclaimed_expert_payload_bytes
            .saturating_sub(u64::try_from(context_reclamation_target_bytes).unwrap_or(u64::MAX));
        let post_reclamation_requirements = ContextAdmissionRequirements {
            current_active_memory_bytes: memory_snapshot_after_reclamation.active_memory_bytes(),
            context_growth_bytes: context_reservation_bytes,
            expert_page_reservation_bytes: maximum_expert_page_reservation_bytes,
            temporary_workspace_bytes: temporary_workspace_reservation_bytes,
            retained_expert_payload_bytes: usize::try_from(
                expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count,
            )
            .unwrap_or(usize::MAX),
            active_memory_ceiling_bytes: configured_mlx_memory_limit_bytes,
            complete_experts_are_resident: false,
        };
        let projected_active_memory_bytes_after_reclamation = post_reclamation_requirements
            .projected_active_memory_bytes()
            .unwrap_or(usize::MAX);
        let post_reclamation_decision = post_reclamation_requirements.decide();
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
                    expert_weight_memory_cache_statistics_after_reclamation
                        .resident_payload_byte_count,
                actual_reclaimed_expert_payload_bytes,
                reclamation_overshoot_bytes,
                expert_eviction_count_delta =
                    expert_weight_memory_cache_statistics_after_reclamation
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
        model.resume_expert_retention_after_request_memory_pressure();
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
            "rejected generation context after retained-expert reclamation remained insufficient"
        );
        return Err(invalid_request_error(
            "generation context exceeds available GPU wired memory",
        ));
    }
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
    Err(invalid_request_error(
        "generation context exceeds available GPU wired memory",
    ))
}

pub(crate) fn context_memory_admission_fits_without_expert_reclamation(
    model: &Qwen3_5Model,
    memory_limits: MlxMemoryLimits,
    context_memory_reservation_bytes_per_token: usize,
    context_token_count_requiring_reservation: usize,
    temporary_workspace_reservation_bytes: usize,
    additional_maximum_expert_page_reservation_bytes: usize,
) -> Result<bool, InferenceEngineError> {
    let projection = build_context_memory_admission_projection(
        model,
        memory_limits,
        context_memory_reservation_bytes_per_token,
        context_token_count_requiring_reservation,
        temporary_workspace_reservation_bytes,
        additional_maximum_expert_page_reservation_bytes,
    )?;
    Ok(matches!(
        ContextAdmissionRequirements {
            current_active_memory_bytes: projection.memory_snapshot.active_memory_bytes(),
            context_growth_bytes: projection.context_reservation_bytes,
            expert_page_reservation_bytes: projection.maximum_expert_page_reservation_bytes,
            temporary_workspace_bytes: temporary_workspace_reservation_bytes,
            retained_expert_payload_bytes: usize::try_from(
                model
                    .expert_weight_memory_cache_statistics()
                    .resident_payload_byte_count,
            )
            .unwrap_or(usize::MAX),
            active_memory_ceiling_bytes: projection.configured_mlx_memory_limit_bytes,
            complete_experts_are_resident: model.resident_expert_weights.is_some(),
        }
        .decide(),
        MemoryAdmissionDecision::Admit
    ))
}

struct ContextMemoryAdmissionProjection {
    memory_snapshot: MlxMemorySnapshot,
    context_reservation_bytes: usize,
    maximum_expert_page_reservation_bytes: usize,
    configured_mlx_memory_limit_bytes: usize,
}

fn build_context_memory_admission_projection(
    model: &Qwen3_5Model,
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
    let maximum_target_expert_page_reservation_bytes = if model.sparse_experts_are_paged() {
        model
            .expert_pager
            .as_ref()
            .map_or(0, |expert_pager| expert_pager.maximum_expert_page_bytes())
    } else {
        0
    };
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
