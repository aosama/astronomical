// Dense.swift — ModelServing.Qwen3_5.Dense
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5/dense/* (the
// dense-variant forward path).
//
// Carried contracts:
// - Dense forward math runs through MLX stock compile-fused layers;
//   custom kernels only with a numerically measured win.
