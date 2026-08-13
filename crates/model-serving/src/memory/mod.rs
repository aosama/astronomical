mod adaptive_ram_growth_guard;
mod expert_memory_admission;
mod mlx_memory_telemetry;
mod mlx_ram_budget;

pub use adaptive_ram_growth_guard::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection,
};
pub use expert_memory_admission::{
    ExpertMemoryAdmissionError, ExpertRetentionReclamationPlan,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
    should_retry_fixed_forward_after_expert_reclamation,
};
pub use mlx_memory_telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};
pub use mlx_ram_budget::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetError,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetPhase, MlxRamBudgetSnapshot,
};
