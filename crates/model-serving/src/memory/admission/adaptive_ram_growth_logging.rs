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
        execution_profile = ?adaptive_ram_growth_context.execution_profile(),
        current_active_memory_bytes = adaptive_ram_growth_projection.current_active_memory_bytes(),
        exact_persistent_growth_bytes =
            adaptive_ram_growth_projection.exact_persistent_growth_bytes(),
        routed_expert_page_reservation_bytes =
            adaptive_ram_growth_projection.routed_expert_page_reservation_bytes(),
        exact_temporary_workspace_bytes =
            adaptive_ram_growth_projection.exact_temporary_workspace_bytes(),
        observed_transient_high_water_bytes =
            adaptive_ram_growth_projection.observed_transient_high_water_bytes(),
        transient_reserve_source = ?adaptive_ram_growth_projection.transient_reserve_source(),
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
