// Normalization.swift — ModelServing.Laguna.Normalization
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/normalization/*
// (family normalization layers).
//
// Carried contracts:
// - Numerically the layers delegate to MLX stock ops; normalization is a
//   hot path, so it stays fused where the stock API allows.
