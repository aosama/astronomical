//! Single owner of memory policy and decisions for every model family.
//!
//! # The ownership contract
//!
//! ```text
//! family code ──measures──▶ memory decides ──decision──▶ family code enacts
//! ```
//!
//! Family modules (`qwen3_5`, `qwen3_5_moe`, `laguna`) measure byte facts from
//! MLX and the artifact, hand them to the modules below, and enact the typed
//! decisions they receive. This package imports nothing from any family; the
//! dependency edge physically cannot point back. A family that re-derives
//! arithmetic owned here is a bug, not a shortcut.
//!
//! # The taxonomy (what a reader must be able to name from this doc alone)
//!
//! - **3 modes** — where expert payload lives right now (`Resident`,
//!   `Hybrid`, `Paged`; classified by `residency/ownership_mode.rs`).
//!   These names are wire-visible and must never gain a second answer.
//! - **1 lifecycle phase** — `phase.rs`: `Prefill`, `GenerationPreparation`,
//!   `Decode`, `Idle`. Budget composition treats `GenerationPreparation` as
//!   `Decode`-equivalent; the growth guard only observes `Prefill`/`Decode`.
//! - **7 layer targets** — `residency` plans one `ExpertLayerResidencyTarget`
//!   per sparse decoder layer (`PreserveComplete`,
//!   `PromoteCompleteOnMandatoryRead`, `PreservePartial`,
//!   `AdmitPartialOnMandatoryRouteRead`, `StreamOperationLocal`,
//!   `ReleasePartial`, `ReleaseCompleteForExactDeficit`).
//! - **2 page classes** — `StableCompleteLayer` and `ElasticRoutedExperts`,
//!   the eviction-priority classes for retained pages.
//!
//! # The naming constitution
//!
//! 1. Every file answers exactly one question, via one suffix: `_admission`
//!    (may this operation begin?), `_budget` (how many bytes does this phase
//!    own or need?), `_reclamation` (what must yield, by how much?),
//!    `_recovery` (what happens after an allocation failure mid-flight?),
//!    `_telemetry` (what did the runtime measure?), `_mode` (where does
//!    expert payload live right now?).
//! 2. The `Mlx` prefix marks true MLX coupling only.
//! 3. One lifecycle phase type (see `phase.rs`).
//! 4. Subpackages group by concern; crate-root re-exports keep consumer paths
//!    stable.
//!
//! # The decision-shape rule
//!
//! - `*Requirements` — exact caller-supplied facts; has `.decide()`.
//! - `*Observation` — sampled facts (runtime-backed); has `.decide()`.
//! - `*Decision` / `*Selection` — typed outcome; execution never re-derives it.
//! - Every rejection carries the named `MemoryBoundary` plus `shortfall_bytes`.

mod admission;
mod budget;
mod ceiling;
mod phase;
mod reclamation;
mod recovery;
mod residency;
mod telemetry;
mod vocabulary;

pub use admission::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyRequirements, ContextAdmissionRequirements, MtpAdmission,
    MtpDepthDowngradeReason, MtpDepthSelection, MtpDraftDepth, MtpDraftDepthError,
    MtpMemoryCandidate, MtpMemoryProjection, MtpMemoryProjectionError, RotatingAdmissionError,
    SpeculativePrefillAdmission, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, request_context_temporary_workspace_bytes,
    retained_expert_payload_capacity_bytes, rotating_committed_token_count,
    rotating_prefill_transient_token_count, safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use admission::{
    log_context_admission_projection, log_generation_context_workspace_reservation,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use budget::context_token_bucket;
pub use budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthProjection, BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES,
    MeasuredExpertLayerPayload, MlxRamBudget, MlxRamBudgetError, MlxRamBudgetMeasurement,
    MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot, RamBudgetGeometryError,
    measured_non_expert_forward_growth_bytes,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
#[cfg(feature = "direct-mlx")]
pub use budget::{MlxAllocationAdmission, MlxAllocationAdmissionError};
pub use ceiling::{MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements};
pub use phase::MemoryPhase;
pub use reclamation::{
    ExpertMemoryAdmissionError, ExpertReclamationPlan,
    complete_residency_exceeds_ceiling_with_activation_headroom,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    projected_active_memory_after_complete_expert_replacement,
    required_complete_residency_activation_headroom_bytes,
    should_retry_fixed_forward_after_expert_reclamation,
};
pub use recovery::{ForwardRecoveryDecision, ForwardRecoveryPolicy, ForwardRecoveryRequirements};
pub use residency::{
    CompleteResidencyHeadroomBoundary, CurrentExpertLayerResidency, ExpertLayerGeometry,
    ExpertLayerResidencyTarget, ExpertResidencyPlan, ExpertResidencyPlanError,
    RequestExpertLayerRole, RequestExpertResidency, RetainedExpertPageClass,
    classify_expert_memory_mode, complete_layer_indexes_required_before_decode,
    hot_expert_warm_slot_count, plan_expert_residency, publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    should_commit_mandatory_complete_layer, should_commit_mandatory_routed_page,
    should_enact_planned_expert_release,
};
pub use telemetry::{MlxActiveMemoryBreakdown, MlxMemoryLimitAdjustment, MlxMemoryTelemetry};
pub use vocabulary::{MemoryAdmissionDecision, MemoryBoundary};
