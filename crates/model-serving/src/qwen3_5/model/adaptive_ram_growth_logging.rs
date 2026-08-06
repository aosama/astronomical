use crate::AdaptiveRamGrowthProjection;
use crate::expert_paging::ExpertWeightMemoryCacheStatistics;

pub(crate) fn log_adaptive_ram_growth_pressure(
    adaptive_ram_growth_projection: &AdaptiveRamGrowthProjection,
    expert_weight_memory_cache_statistics_before_reclamation: ExpertWeightMemoryCacheStatistics,
    expert_weight_memory_cache_statistics_after_reclamation: ExpertWeightMemoryCacheStatistics,
    allocator_cache_memory_bytes_observed: usize,
    required_reclamation_bytes: usize,
    action: &'static str,
) {
    let retained_expert_payload_bytes_before =
        expert_weight_memory_cache_statistics_before_reclamation.resident_payload_byte_count;
    let retained_expert_payload_bytes_after =
        expert_weight_memory_cache_statistics_after_reclamation.resident_payload_byte_count;
    let actual_reclaimed_expert_payload_bytes =
        retained_expert_payload_bytes_before.saturating_sub(retained_expert_payload_bytes_after);
    let required_reclamation_bytes_u64 =
        u64::try_from(required_reclamation_bytes).unwrap_or(u64::MAX);
    let reclamation_overshoot_bytes =
        actual_reclaimed_expert_payload_bytes.saturating_sub(required_reclamation_bytes_u64);
    tracing::info!(
        action,
        current_active_memory_bytes = adaptive_ram_growth_projection.current_active_memory_bytes(),
        exact_persistent_growth_bytes =
            adaptive_ram_growth_projection.exact_persistent_growth_bytes(),
        observed_transient_high_water_bytes =
            adaptive_ram_growth_projection.observed_transient_high_water_bytes(),
        stable_projected_bytes = adaptive_ram_growth_projection.stable_projected_bytes(),
        peak_projected_bytes = adaptive_ram_growth_projection.peak_projected_bytes(),
        soft_recovery_projected_bytes =
            adaptive_ram_growth_projection.soft_recovery_projected_bytes(),
        required_reclamation_bytes = adaptive_ram_growth_projection.required_reclamation_bytes(),
        soft_reserve_shortfall_bytes =
            adaptive_ram_growth_projection.soft_reserve_shortfall_bytes(),
        active_memory_limit_bytes = adaptive_ram_growth_projection.active_memory_limit_bytes(),
        allowed_active_memory_bytes = adaptive_ram_growth_projection.allowed_active_memory_bytes(),
        required_reclamation_bytes,
        retained_expert_payload_bytes_before,
        retained_expert_payload_bytes_after,
        retained_partial_expert_count_before =
            expert_weight_memory_cache_statistics_before_reclamation
                .entry_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_before_reclamation.complete_layer_count,
                ),
        retained_partial_expert_count_after =
            expert_weight_memory_cache_statistics_after_reclamation
                .entry_count
                .saturating_sub(
                    expert_weight_memory_cache_statistics_after_reclamation.complete_layer_count,
                ),
        retained_complete_layer_count_before =
            expert_weight_memory_cache_statistics_before_reclamation.complete_layer_count,
        retained_complete_layer_count_after =
            expert_weight_memory_cache_statistics_after_reclamation.complete_layer_count,
        actual_reclaimed_expert_payload_bytes,
        reclamation_overshoot_bytes,
        allocator_cache_memory_bytes_observed,
        "adaptive RAM growth pressure decision"
    );
}
