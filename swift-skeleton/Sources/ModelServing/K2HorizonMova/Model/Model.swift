// Model.swift — ModelServing.K2HorizonMova.Model
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/k2_horizon_mova/model/*
// including its fused_expert_decode submodule (weight layout and forward
// math, fused expert decode).
//
// Carried contracts:
// - Fused expert decode engages only after KernelCapability probes verify
//   the target — never silently.
// - Expert memory questions route through the Memory subpackage; no
//   family-local memory arithmetic.
