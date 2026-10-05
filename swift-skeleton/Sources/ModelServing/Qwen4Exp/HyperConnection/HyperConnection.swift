// HyperConnection.swift — ModelServing.Qwen4Exp.HyperConnection
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen4_exp/
// hyper_connection/* (the family's hyper-connection mixing layers).
//
// Carried contracts:
// - Mixing math runs through MLX stock operations; GPU (Graphics
//   Processing Unit) dispatches are combined wherever the stock API
//   allows.
