//! Single owner of memory policy and decisions for every model family.
//!
//! # The ownership contract
//!
//! ```text
//! family code ──measures──▶ memory decides ──decision──▶ family code enacts
//! ```
//!
//! Family modules (`qwen3_5`, `qwen3_5_streaming`) measure byte facts from
//! MLX and the artifact, hand them to the pure arithmetic in
//! `astronomical-memory-policy` (re-exported below) and to the MLX-facing
//! owners in this package, and enact the typed decisions they receive. This
//! package imports nothing from any family; the dependency edge physically
//! cannot point back. A family that re-derives arithmetic owned here is a
//! bug, not a shortcut.
//!
//! # The taxonomy (what a reader must be able to name from this doc alone)
//!
//! - **3 modes** — where expert payload lives right now (`Resident`,
//!   `Hybrid`, `Paged`; classified by `residency/ownership_mode.rs`).
//!   These names are wire-visible and must never gain a second answer.
//! - **1 lifecycle phase** — `MemoryPhase` (in `astronomical-memory-policy`,
//!   re-exported here): `Prefill`, `GenerationPreparation`, `Decode`,
//!   `Idle`. The residency planner treats `GenerationPreparation` like
//!   `Decode`.
//! - **1 owner of the limit** — the memory governor
//!   (`astronomical-memory-policy::ledger`). Each engine opens one, tells
//!   it what happens and applies the capacity it answers with.
//! - **3 layer states** — when a reply begins, `residency` plans one
//!   `ExpertLayerState` per sparse decoder layer: `Kept`, `Hot` or
//!   `Absent`. What to do about a layer follows from its state and from
//!   what the astro expert cache holds.
//! - **2 page classes** — `StableCompleteLayer` and `ElasticRoutedExperts`,
//!   the eviction-priority classes for retained *prefill* pages.
//!
//! # The naming constitution
//!
//! 1. Every file answers exactly one question, via one suffix: `_admission`
//!    (may this operation begin?), `_telemetry` (what did the runtime
//!    measure?), `_mode` (where does expert payload live right now?).
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
#[cfg(any(test, feature = "direct-mlx"))]
pub(crate) mod decode_seating;
#[cfg(test)]
mod decode_seating_tests;
// The dense-resident ceiling policy is consumed only by the direct-MLX
// engine, so an isolated default-feature build would otherwise flag the
// whole module as dead (workspace builds unify the feature away).
mod astro_expert_cache_arithmetic;
#[cfg(feature = "direct-mlx")]
mod dense_resident;
// Only the direct-MLX streaming engine reads it; the unit tests read it
// without the graphics-processor runtime.
#[cfg(any(feature = "direct-mlx", test))]
pub(crate) mod macos_wired_memory;
// Only the direct-MLX streaming engine records replies through it; the unit
// tests own it without the graphics-processor runtime.
#[cfg(any(feature = "direct-mlx", test))]
pub(crate) mod mlx_buffer_cache_reply_record;
// Only the direct-MLX streaming engine records replies through it; the unit
// tests own it without the graphics-processor runtime.
#[cfg(any(feature = "direct-mlx", test))]
pub(crate) mod reply_expert_route_record;
#[cfg(feature = "direct-mlx")]
pub use reply_expert_route_record::SessionExpertRouteSample;
// The telemetry and ceiling-utilization records are plain values, so they
// live in the memory policy crate (issue #2340) where engines below this
// crate can fill them in; the alias keeps `crate::memory::telemetry` paths.
pub(crate) use astronomical_memory_policy::telemetry;

pub use admission::{
    AllocationAdmissionDecision, AllocationAdmissionObservation, CompleteResidencyDecision,
    CompleteResidencyRequirements, RetainedPrefixRetentionDecision,
    RetainedPrefixRetentionRequirements, RotatingAdmissionError, VISIBLE_ANSWER_RESERVE_TOKENS,
    bounded_output_reservation_token_count, persistent_context_restore_workspace_bytes,
    rotating_committed_token_count, rotating_prefill_transient_token_count,
    safe_minimum_active_memory_ceiling_bytes,
};
pub use astronomical_memory_policy::ceiling::{
    MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements,
};
pub use budget::{
    MeasuredExpertLayerPayload, MemoryReserveSnapshot, ModelMemoryFacts, ModelMemoryFactsError,
    ModelMemoryGeometry, ModelMemoryGeometryError, model_memory_geometry_from_measured_layer_facts,
};
#[cfg(feature = "direct-mlx")]
pub use budget::{MlxAllocationAdmission, MlxAllocationAdmissionError};
#[cfg(feature = "direct-mlx")]
pub(crate) use dense_resident::dense_resident_safe_minimum_mlx_memory_ceiling_bytes;
// Byte-arithmetic page-peaking is pure planning over validated shapes; the
// direct-MLX admission callers and the unit artifact-contract tests share
// it, while the plain default-feature lib omits the module entirely.
#[cfg(any(feature = "direct-mlx", test))]
pub(crate) use astro_expert_cache_arithmetic::expert_page_peak_allocation_bytes;
#[cfg(feature = "direct-mlx")]
pub(crate) use astro_expert_cache_arithmetic::source_expert_page_payload_bytes;
#[cfg(feature = "direct-mlx")]
pub(crate) use astro_expert_cache_arithmetic::{
    ExpertStreamingSessionCounters, RouteCoverageClass, StreamedPageSourcePlan,
    expert_layer_topology_from_residency, no_astro_expert_cache_route_source_bytes,
    record_streamed_page_source_plan_in_both_ledgers,
};
pub use astronomical_memory_policy::forward_admission::{
    ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection, StreamingEngineRequirement,
};
pub use astronomical_memory_policy::geometry_floor::geometry_safe_minimum_mlx_memory_ceiling_bytes;
pub use astronomical_memory_policy::phase::MemoryPhase;
pub use astronomical_memory_policy::placement::{
    EnginePlacement, ResidentPromotionBlocker, ResidentPromotionDecision,
    should_promote_demoted_streaming_engine_to_resident,
};
pub use astronomical_memory_policy::reclamation::ExpertMemoryAdmissionError;
pub use astronomical_memory_policy::residency::seated_complete_layer_count;
// The direct-MLX streaming planner (qwen3_5_streaming's gated
// phase_aware_expert_residency) imports residency by module path
// (`crate::memory::residency::ExpertLayerGeometry`), so the module stays
// nameable here under the same feature gate, without widening the crate's
// public surface.
#[cfg(feature = "direct-mlx")]
pub(crate) use astronomical_memory_policy::residency;
pub use astronomical_memory_policy::residency::{
    CompleteResidencyHeadroomBoundary, CurrentExpertLayerResidency, ExpertLayerState,
    ExpertResidencyPlan, ExpertResidencyPlanError, RetainedExpertPageClass,
    StreamingStartupResidencyFacts, classify_expert_memory_mode, complete_layers_beyond_keep_count,
    decode_warm_floor_eviction_indexes, decode_whole_layer_keep_count, hot_expert_warm_slot_count,
    kept_layer_indexes_to_seat_before_reply, minimum_decode_warm_slot_count,
    observed_expert_memory_mode, plan_expert_residency, should_commit_mandatory_complete_layer,
    should_commit_mandatory_routed_page, should_commit_multi_token_union_as_complete_layer,
    should_complete_partial_layer_on_prefill_read, should_seed_partial_layer_on_prefill_read,
};
pub use astronomical_memory_policy::serving_ceiling::{
    effective_mlx_memory_ceiling_after_wired_claim, effective_mlx_memory_ceiling_bytes,
    floor_raise_limit_bytes, floor_raised_ceiling_bytes, machine_ceiling_from_physical_ram_bytes,
};
pub use astronomical_memory_policy::utilization::MemoryCeilingUtilization;
pub use astronomical_memory_policy::vocabulary::MemoryBoundary;
