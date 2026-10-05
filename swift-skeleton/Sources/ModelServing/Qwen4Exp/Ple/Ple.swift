// Ple.swift — ModelServing.Qwen4Exp.Ple
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen4_exp/ple/*
// (parameter-efficient expert layers).
//
// Carried contracts:
// - Layer math runs through MLX stock compile-fused layers; custom
//   kernels only with a numerically measured win.
