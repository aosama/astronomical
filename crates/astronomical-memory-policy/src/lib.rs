//! Pure memory policy and expert-paging arithmetic, published from its own
//! workspace crate so a policy change never rebuilds a model family.
//!
//! This crate is the memory-management home the `AGENTS.md` ownership rule
//! names ("all memory management code lives in the memory package"). It is a
//! deliberate dependency-graph fact: it depends on nothing graphics-processor
//! related, so its tests build and run in the offline lane in a fraction of a
//! second and a budget or residency change never pulls in the MLX bindings.
//!
//! The layering contract (`verification/tooling_contracts/
//! test-memory-policy-layering-contract.sh`) fails if a MLX dependency is
//! reintroduced here, so the boundary is enforced by the dependency graph
//! instead of by convention. The MLX-facing owners (retained cache slot
//! tables, streaming readers, the direct-MLX admission types) stay in
//! `astronomical-model-serving` and depend on this crate; the dependency
//! direction is one way.

pub mod serving_ceiling;

#[cfg(test)]
mod serving_ceiling_tests;
