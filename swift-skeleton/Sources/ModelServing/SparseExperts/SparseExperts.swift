// SparseExperts.swift — ModelServing.SparseExperts
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/sparse_experts/* —
// assignment permutation and sort, gathered projection, weighted sum:
// Mixture of Experts expert selection and routing math.
//
// Carried contracts:
// - Expert paging timings feed the switchable performance attribution log
//   (expert paging is a first-class critical path).
// - MoE (Mixture of Experts) shapes come from config-driven structural
//   checks (expert counts, group sizes), not hardcoded per-variant tables.
