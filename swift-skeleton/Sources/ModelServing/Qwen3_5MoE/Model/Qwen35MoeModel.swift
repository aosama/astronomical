// Qwen35MoeModel.swift — ModelServing.Qwen3_5MoE.Model
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5_moe/model/* —
// routing, paged_execution, resident_execution, mixed_decode_execution,
// feed_forward_weights, phase_aware_expert_residency, and neighbors.
//
// Carried contracts:
// - Execution modes (resident, paged, mixed) are selected from the
//   request's memory plan, never guessed per layer.
// - Route observation stays available for attribution so paging decisions
//   are explainable after the fact.
