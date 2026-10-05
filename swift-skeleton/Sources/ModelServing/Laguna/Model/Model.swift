// Model.swift — ModelServing.Laguna.Model
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/model/* (weight
// layout and forward math).
//
// Carried contracts:
// - Forward math runs through MLX stock compile-fused layers; custom
//   kernels only with a numerically measured win.
