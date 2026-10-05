// ExpertResidency.swift — ModelServing.Qwen3_5MoE.ExpertResidency
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5_moe/
// expert_residency/* — resident_expert_weights, resident_expert_loading,
// resident_expert_layer_weights, resident_gate_up_fusion.
//
// Carried contracts:
// - Residency transitions (which experts sit in wired memory) are
//   decided by the Memory subpackage; this module executes them.
