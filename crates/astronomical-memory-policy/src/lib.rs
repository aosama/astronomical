//! Pure memory policy and expert-paging arithmetic, published from its own
//! workspace crate so a policy change never rebuilds a model family.
//!
//! This crate is the memory-management home the `AGENTS.md` ownership rule
//! names ("all memory management code lives in the memory package"). It is a
//! deliberate dependency-graph fact: it depends on nothing graphics-processor
//! related, so its tests build and run in the offline lane in a fraction of a
//! second and a budget or residency change never pulls in the MLX bindings.
//!
//! The layering contract (`cargo xtask check-architecture`) fails if a MLX
//! dependency is
//! reintroduced here, so the boundary is enforced by the dependency graph
//! instead of by convention. The MLX-facing owners (retained cache slot
//! tables, streaming readers, the runtime-backed allocation admission, the
//! direct-MLX admission logging) stay in `astronomical-model-serving` and
//! depend on this crate; the dependency direction is one way.
//!
//! # Module map
//!
//! - `admission` — may this operation begin (`AllocationAdmissionDecision`,
//!   `CompleteResidencyDecision`, context workspaces, retained-prefix
//!   retention, rotating admission).
//! - `expert_paging` — pure expert-paging support arithmetic: the
//!   safetensors header parsing (`safetensors_header`, over `safetensors`
//!   and `strict_json`) and the expert weight memory-cache statistics. The
//!   quantized expert manifests stay in `astronomical-model-serving` — they
//!   embed model-family configuration types.
//! - `phase` — the one lifecycle phase type (`MemoryPhase`).
//! - `vocabulary` — the shared decision vocabulary (`MemoryBoundary`,
//!   `MemoryAdmissionDecision`): every rejection names its boundary.
//! - `reclamation` — what must yield, by how much (`ExpertReclamationPlan`,
//!   the paged step enum `PagedExpertReclamationStep`, the admission error
//!   vocabulary `ExpertMemoryAdmissionError`, and the reclamation
//!   arithmetic).
//! - `budget` — how many bytes this phase owns or needs (`MlxRamBudget`
//!   composition, learned activation evidence, adaptive growth guards,
//!   measured model geometry).
//! - `ceiling` — may the worker raise or lower the live ceiling right now,
//!   and what ownership must move first.
//! - `geometry_floor` — the smallest servable ceiling derived from validated
//!   artifact geometry, independent of the ceiling that produced the load.
//! - `recovery` — what happens after an allocation failure mid-forward:
//!   which retry is legal and what workspace it needs.
//! - `residency` — where each sparse layer's experts live for one request:
//!   layer targets, page classes, seating plans, foundation overlay, and
//!   plan validation.
//! - `serving_ceiling` — resolution of the effective MLX memory ceiling from
//!   configuration and the machine bound.

pub mod admission;
pub mod budget;
pub mod ceiling;
pub mod expert_paging;
pub mod geometry_floor;
pub mod phase;
pub mod reclamation;
pub mod recovery;
pub mod residency;
pub mod safetensors;
pub mod serving_ceiling;
pub mod strict_json;
pub mod vocabulary;

#[cfg(test)]
mod adaptive_ram_growth_execution_profile_tests;
#[cfg(test)]
mod adaptive_ram_growth_guard_admission_tests;
#[cfg(test)]
mod adaptive_ram_growth_guard_reclamation_tests;
#[cfg(test)]
mod adaptive_ram_growth_observations_tests;
#[cfg(test)]
mod ceiling_boundary_tests;
#[cfg(test)]
mod memory_policy_tests;
#[cfg(test)]
mod reclamation_boundary_tests;
#[cfg(test)]
mod serving_ceiling_tests;
#[cfg(test)]
mod strict_json_boundary_tests;
