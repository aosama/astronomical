mod adaptive_ram_growth_guard;
mod allocation_admission;
mod ceiling_change;
mod complete_residency_headroom_boundary;
mod context_admission;
#[cfg(feature = "direct-mlx")]
mod context_admission_logging;
mod decision;
mod decode_complete_layer_seating;
mod expert_memory_admission;
mod expert_ownership_mode;
mod expert_residency;
mod forward_recovery;
#[cfg(feature = "direct-mlx")]
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
pub use complete_residency_headroom_boundary::CompleteResidencyHeadroomBoundary;
pub use context_admission::{
    ContextAdmissionRequirements, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, request_context_temporary_workspace_bytes,
    safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use context_admission_logging::{
    log_context_admission_projection, log_generation_context_workspace_reservation,
};
pub use decision::{MemoryAdmissionDecision, MemoryBoundary};
pub use decode_complete_layer_seating::complete_layer_indexes_required_before_decode;
pub use expert_memory_admission::{
    ExpertMemoryAdmissionError, ExpertRetentionReclamationPlan,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
    should_retry_fixed_forward_after_expert_reclamation,
};
pub use expert_ownership_mode::classify_expert_memory_mode;
pub use expert_residency::{
    CurrentExpertLayerResidency, ExpertLayerGeometry, ExpertLayerResidencyTarget,
    ExpertResidencyPhase, PhaseAwareExpertResidencyPlan, PhaseAwareExpertResidencyPlanError,
    RequestExpertLayerRole, RequestExpertResidency, RetainedExpertPageClass,
    plan_phase_aware_expert_residency, publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_enact_planned_expert_release,
};
pub use forward_recovery::{
    ForwardRecoveryDecision, ForwardRecoveryPolicy, ForwardRecoveryRequirements,
};
#[cfg(feature = "direct-mlx")]
pub use live_allocation_budget::{MlxAllocationBudget, MlxAllocationBudgetError};
pub use mlx_memory_telemetry::{
    MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use mlx_ram_budget::context_token_bucket;
pub use mlx_ram_budget::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetError,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetPhase, MlxRamBudgetSnapshot,
    measured_non_expert_forward_growth_bytes,
};
pub use residency_admission::{CompleteResidencyDecision, CompleteResidencyRequirements};
pub use speculative_prefill_admission::SpeculativePrefillAdmission;
