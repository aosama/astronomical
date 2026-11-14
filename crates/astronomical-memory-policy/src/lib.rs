//! Pure memory policy and expert-paging arithmetic, published from its own
//! workspace crate so a policy change never rebuilds a model family.
//!
//! This crate is the memory-management home the `AGENTS.md` ownership rule
//! names ("all memory management code lives in the memory package"). It is a
//! deliberate dependency-graph fact: it depends on nothing graphics-processor
//! related, so its tests build and run in the integration lane in a fraction of a
//! second and a ledger or residency change never pulls in the MLX bindings.
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
//! - `vocabulary` — the shared decision vocabulary (`MemoryBoundary`):
//!   every rejection names its boundary.
//! - `reclamation` — the checked arithmetic for replacing kept experts with
//!   a complete set at load, and its error vocabulary
//!   (`ExpertMemoryAdmissionError`).
//! - `forward_admission` — the in-memory engine's typed answer for one
//!   forward: run it, or say why the request moves to the streaming engine.
//! - `placement` — the typed engine-placement lifecycle: startup selection,
//!   request demotion with its typed cause, and the bounded re-promotion
//!   gate back to the resident engine.
//! - `model_memory` — the facts a memory governor is opened with: the limit
//!   a model runs under and its measured geometry (`ModelMemoryFacts`), and
//!   the snapshot the status breakdown is composed against.
//! - `ceiling` — may the worker raise or lower the live ceiling right now,
//!   and what ownership must move first.
//! - `ledger` — the one owner of the whole limit: named accounts that always
//!   add up, with the kept-weight capacity as the remainder (issue #2221).
//!   Its tests include `simulation`, a scripted engine that replays whole
//!   requests against the ledger without a model.
//! - `geometry_floor` — the smallest servable ceiling derived from validated
//!   artifact geometry, independent of the ceiling that produced the load.
//! - `residency` — where each sparse layer's experts live for one request:
//!   layer targets, page classes, seating plans, foundation overlay, and
//!   plan validation.
//! - `serving_ceiling` — resolution of the effective MLX memory ceiling from
//!   configuration and the machine bound.

pub mod admission;
pub mod ceiling;
pub mod expert_paging;
pub mod forward_admission;
pub mod geometry_floor;
pub mod ledger;
pub mod model_memory;
pub mod neural_engine_admission;
pub mod neural_engine_program_memory_watch;
pub mod phase;
pub mod placement;
pub mod reclamation;
pub mod residency;
pub mod safetensors;
pub mod serving_ceiling;
pub mod strict_json;
pub mod vocabulary;

pub use forward_admission::{
    ResidentForwardAdmissionDecision, ResidentForwardAdmissionRejection, StreamingEngineRequirement,
};
pub use placement::{
    EnginePlacement, ResidentPromotionBlocker, ResidentPromotionDecision,
    should_promote_demoted_streaming_engine_to_resident,
};
pub use residency::observed_expert_memory_mode;

#[cfg(test)]
mod simulation;

#[cfg(test)]
mod ceiling_boundary_tests;
#[cfg(test)]
mod memory_policy_tests;
#[cfg(test)]
mod serving_ceiling_tests;
#[cfg(test)]
mod strict_json_boundary_tests;
