// MlxRuntime.swift — RuntimeIntegration.MlxRuntime
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/runtime-integration/src/mlx_runtime/* —
// error_handling, initialization, memory_policy, metallib, safetensors.
// This crate is the policy owner for MLX-backed execution and replaces the
// retired crates/mlx-c-rust binding surface with MLX-Swift.
//
// Carried contracts:
// - Prefer MLX's stock compile-fused layers; a custom implementation ships
//   only with a numerically measured win over the stock layer.
// - Minimize GPU (Graphics Processing Unit) dispatches; combine operations
//   into single dispatches wherever possible.
// - Performance attribution logging is switchable through configuration and
//   captures start and end time per operation on the serving critical path.
