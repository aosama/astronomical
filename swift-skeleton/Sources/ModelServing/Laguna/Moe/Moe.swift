// Moe.swift — ModelServing.Laguna.Moe
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/moe/* (Mixture
// of Experts routing for the family).
//
// Carried contracts:
// - MoE shapes come from config-driven structural checks (expert counts,
//   group sizes), not hardcoded per-variant tables.
