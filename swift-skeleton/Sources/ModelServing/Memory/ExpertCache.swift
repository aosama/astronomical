// ExpertCache.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory expert cache
// (paged expert weights for Mixture of Experts models).
//
// Carried contracts:
// - Expert paging timings feed the switchable performance attribution log
//   (expert paging is a first-class critical path).
// - Cache sizing derives from the memory budget, never from machine-specific
//   constants.
