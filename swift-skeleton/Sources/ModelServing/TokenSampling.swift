// TokenSampling.swift — ModelServing
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving token sampling (the sampler
// surface shared by all text families).
//
// Carried contracts:
// - Temperature is never forced to 0 in tests or API defaults; the provider
//   determines it.
// - Sampling math runs through MLX stock operations; custom kernels only
//   with a numerically measured win.
