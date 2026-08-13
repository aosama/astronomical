mod adaptive_ram_growth_guard;
mod allocation_admission;
mod ceiling_change;
mod context_admission;
mod decision;
mod expert_memory_admission;
mod forward_recovery;
mod live_allocation_budget;
mod mlx_memory_telemetry;
mod mlx_ram_budget;
mod residency_admission;
mod speculative_prefill_admission;

pub use allocation_admission::{
    AllocationAdmissionDecision, AllocationAdmissionObservation,
    retained_expert_payload_capacity_bytes,
};

pub use adaptive_ram_growth_guard::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase, AdaptiveRamGrowthProjection,
};
pub use ceiling_change::{MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements};
pub use context_admission::{
    ContextAdmissionRequirements, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, safe_minimum_active_memory_ceiling_bytes,
};
pub use decision::{MemoryAdmissionDecision, MemoryBoundary};
pub use expert_memory_admission::{
    ExpertMemoryAdmissionError, ExpertRetentionReclamationPlan,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
    should_retry_fixed_forward_after_expert_reclamation,
};
pub use forward_recovery::{
    ForwardRecoveryDecision, ForwardRecoveryPolicy, ForwardRecoveryRequirements,
};
pub use live_allocation_budget::{MlxAllocationBudget, MlxAllocationBudgetError};
pub use mlx_memory_telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};
pub use mlx_ram_budget::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetError,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetPhase, MlxRamBudgetSnapshot,
};
pub use residency_admission::{CompleteResidencyDecision, CompleteResidencyRequirements};
pub use speculative_prefill_admission::SpeculativePrefillAdmission;
