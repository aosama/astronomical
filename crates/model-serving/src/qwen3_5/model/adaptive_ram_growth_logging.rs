use crate::expert_paging::ExpertWeightMemoryCacheStatistics;
use crate::{AdaptiveRamGrowthContext, AdaptiveRamGrowthProjection};

pub(crate) fn log_adaptive_ram_growth_admission_decision(
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    adaptive_ram_growth_projection: &AdaptiveRamGrowthProjection,
    decision: &'static str,
) {
    let stable_limit_exceeded = adaptive_ram_growth_projection.stable_projected_bytes()
        > adaptive_ram_growth_projection.active_memory_ceiling_bytes();
    let peak_limit_exceeded = adaptive_ram_growth_projection.peak_projected_bytes()
        > adaptive_ram_growth_projection.allowed_active_memory_bytes();
    let recovery_reserve_exceeded = adaptive_ram_growth_projection.recovery_projected_bytes()
        > adaptive_ram_growth_projection.allowed_active_memory_bytes();
    let recovery_reserve_only_trigger =
        !stable_limit_exceeded && !peak_limit_exceeded && recovery_reserve_exceeded;
    tracing::info!(
        decision,
        phase = ?adaptive_ram_growth_context.memory_phase(),
        forward_token_count = adaptive_ram_growth_context.forward_token_count(),
        sparse_experts_are_paged = adaptive_ram_growth_context.sparse_experts_are_paged(),
        current_active_memory_bytes = adaptive_ram_growth_projection.current_active_memory_bytes(),
        exact_persistent_growth_bytes =
            adaptive_ram_growth_projection.exact_persistent_growth_bytes(),
        routed_expert_page_reservation_bytes =
            adaptive_ram_growth_projection.routed_expert_page_reservation_bytes(),
        exact_temporary_workspace_bytes =
            adaptive_ram_growth_projection.exact_temporary_workspace_bytes(),
        observed_transient_high_water_bytes =
            adaptive_ram_growth_projection.observed_transient_high_water_bytes(),
        stable_projected_bytes = adaptive_ram_growth_projection.stable_projected_bytes(),
        peak_projected_bytes = adaptive_ram_growth_projection.peak_projected_bytes(),
        recovery_projected_bytes = adaptive_ram_growth_projection.recovery_projected_bytes(),
        active_memory_ceiling_bytes = adaptive_ram_growth_projection.active_memory_ceiling_bytes(),
        allowed_active_memory_bytes = adaptive_ram_growth_projection.allowed_active_memory_bytes(),
        operation_reclamation_required_bytes =
            adaptive_ram_growth_projection.operation_reclamation_required_bytes(),
        recovery_reserve_shortfall_bytes =
            adaptive_ram_growth_projection.recovery_reserve_shortfall_bytes(),
        stable_limit_exceeded,
        peak_limit_exceeded,
        recovery_reserve_exceeded,
        recovery_reserve_only_trigger,
        "adaptive RAM growth admission decision"
    );
}

pub(crate) fn log_adaptive_ram_growth_pressure(
    adaptive_ram_growth_projection: &AdaptiveRamGrowthProjection,
    expert_weight_memory_cache_statistics_before_reclamation: ExpertWeightMemoryCacheStatistics,
    expert_weight_memory_cache_statistics_after_reclamation: ExpertWeightMemoryCacheStatistics,
    allocator_cache_memory_bytes_observed: usize,
    expert_reclamation_target_bytes: usize,
    action: &'static str,
) {
    let retained_expert_payload_bytes_before =
        expert_weight_memory_cache_statistics_before_reclamation.resident_payload_byte_count;
    let retained_expert_payload_bytes_after =
        expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count;
    let actual_reclaimed_expert_payload_bytes =
        retained_expert_payload_bytes_before.saturating_sub(retained_expert_payload_bytes_after);
    let expert_reclamation_target_bytes_u64 =
        u64::try_from(expert_reclamation_target_bytes).unwrap_or(u64::MAX);
    let reclamation_overshoot_bytes =
        actual_reclaimed_expert_payload_bytes.saturating_sub(expert_reclamation_target_bytes_u64);
    tracing::info!(
        action,
        current_active_memory_bytes = adaptive_ram_growth_projection.current_active_memory_bytes(),
        exact_persistent_growth_bytes =
            adaptive_ram_growth_projection.exact_persistent_growth_bytes(),
        routed_expert_page_reservation_bytes =
            adaptive_ram_growth_projection.routed_expert_page_reservation_bytes(),
        observed_transient_high_water_bytes =
            adaptive_ram_growth_projection.observed_transient_high_water_bytes(),
        stable_projected_bytes = adaptive_ram_growth_projection.stable_projected_bytes(),
        peak_projected_bytes = adaptive_ram_growth_projection.peak_projected_bytes(),
        recovery_projected_bytes = adaptive_ram_growth_projection.recovery_projected_bytes(),
        operation_reclamation_required_bytes =
            adaptive_ram_growth_projection.operation_reclamation_required_bytes(),
        recovery_reserve_shortfall_bytes =
            adaptive_ram_growth_projection.recovery_reserve_shortfall_bytes(),
        active_memory_ceiling_bytes = adaptive_ram_growth_projection.active_memory_ceiling_bytes(),
        allowed_active_memory_bytes = adaptive_ram_growth_projection.allowed_active_memory_bytes(),
        expert_reclamation_target_bytes,
        retained_expert_payload_bytes_before,
        retained_expert_payload_bytes_after,
        retained_one_expert_page_count_before =
            expert_weight_memory_cache_statistics_before_reclamation.entry_count,
        retained_one_expert_page_count_after =
            expert_weight_memory_cache_statistics_after_reclamation.entry_count,
        actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes,
        allocator_cache_memory_bytes_observed,
        "adaptive RAM growth pressure decision"
    );
}
