use astronomical_runtime_integration::MlxMemoryLimits;

use crate::InferenceEngineError;

use super::super::inference_execution::qwen3_5_runtime_error;
use super::Qwen3_5Model;
use crate::qwen3_5_moe::reclaim_retained_experts_for_request_memory_pressure;

/// Rejects a context before request cache allocation when its model-derived
/// reservation would cross the machine-derived request admission budget.
pub(crate) fn validate_context_memory_admission(
    model: &Qwen3_5Model,
    memory_limits: MlxMemoryLimits,
    context_memory_reservation_bytes_per_token: usize,
    context_token_count_requiring_reservation: usize,
    temporary_workspace_reservation_bytes: usize,
    additional_maximum_expert_page_reservation_bytes: usize,
) -> Result<(), InferenceEngineError> {
    let initial_memory_snapshot = model
        .runtime()
        .memory_snapshot()
        .map_err(qwen3_5_runtime_error)?;
    let context_reservation_bytes = context_token_count_requiring_reservation
        .checked_mul(context_memory_reservation_bytes_per_token)
        .ok_or_else(|| invalid_request_error("generation context memory reservation overflowed"))?;
    let maximum_expert_page_reservation_bytes = model
        .expert_pager
        .as_ref()
        .map_or(0, |expert_pager| expert_pager.maximum_expert_page_bytes());
    let maximum_expert_page_reservation_bytes =
        usize::try_from(maximum_expert_page_reservation_bytes)
            .map_err(|_| {
                invalid_request_error("maximum expert page reservation exceeds the platform range")
            })?
            .checked_add(additional_maximum_expert_page_reservation_bytes)
            .ok_or_else(|| invalid_request_error("combined expert page reservation overflowed"))?;
    let initial_projected_active_memory_bytes =
        context_memory_admission_projected_active_memory_bytes(
            initial_memory_snapshot.active_memory_bytes(),
            context_reservation_bytes,
            maximum_expert_page_reservation_bytes,
        )
        .and_then(|projected_active_memory_bytes| {
            projected_active_memory_bytes.checked_add(temporary_workspace_reservation_bytes)
        })
        .ok_or_else(|| invalid_request_error("generation memory projection overflowed"))?;
    let configured_mlx_memory_limit_bytes = memory_limits.active_memory_limit_bytes();
    if initial_projected_active_memory_bytes <= configured_mlx_memory_limit_bytes {
        return Ok(());
    }
    let context_reclamation_target_bytes =
        initial_projected_active_memory_bytes.saturating_sub(configured_mlx_memory_limit_bytes);
    let expert_weight_memory_cache_statistics_before_reclamation =
        model.expert_weight_memory_cache_statistics();
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
        let projected_active_memory_bytes_after_reclamation =
            context_memory_admission_projected_active_memory_bytes(
                memory_snapshot_after_reclamation.active_memory_bytes(),
                context_reservation_bytes,
                maximum_expert_page_reservation_bytes,
            )
            .and_then(|projected_active_memory_bytes| {
                projected_active_memory_bytes.checked_add(temporary_workspace_reservation_bytes)
            })
            .ok_or_else(|| invalid_request_error("generation memory projection overflowed"))?;
        if projected_active_memory_bytes_after_reclamation <= configured_mlx_memory_limit_bytes {
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

/// Returns the duplicated full-attention key/value payload retained while
/// persistent prompt-cache blocks are concatenated into live decoder state.
#[must_use]
pub fn persistent_prompt_cache_restore_temporary_workspace_bytes(
    context_memory_reservation_bytes_per_token: usize,
    restored_persistent_prompt_cache_token_count: usize,
) -> Option<usize> {
    context_memory_reservation_bytes_per_token
        .checked_mul(restored_persistent_prompt_cache_token_count)
}

/// Projects request-active memory from exact persistent growth and expert-page ownership.
#[must_use]
pub fn context_memory_admission_projected_active_memory_bytes(
    current_active_memory_bytes: usize,
    context_reservation_bytes: usize,
    maximum_expert_page_reservation_bytes: usize,
) -> Option<usize> {
    current_active_memory_bytes
        .checked_add(context_reservation_bytes)?
        .checked_add(maximum_expert_page_reservation_bytes)
}

/// Combines exact target and additional persistent growth without adding headroom.
#[must_use]
pub fn combined_target_and_additional_persistent_growth_bytes(
    target_persistent_state_growth_bytes: usize,
    additional_persistent_state_growth_bytes: usize,
) -> Result<usize, InferenceEngineError> {
    target_persistent_state_growth_bytes
        .checked_add(additional_persistent_state_growth_bytes)
        .ok_or_else(|| invalid_request_error("target and additional persistent growth overflowed"))
}

pub(crate) fn invalid_request_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.into(),
    }
}
