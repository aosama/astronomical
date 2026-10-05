// Qwen35MoeExpertPaging.swift — ModelServing.Qwen3_5MoE.ExpertPaging
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5_moe/
// expert_paging/* — expert_pager (+rust_expert_streaming),
// expert_pager_construction, paged_expert_weights,
// quantized_expert_layer_plan, retained_expert_cache (+demand, flush,
// insert, memory_behavior, previous_token_prefetch, reclamation,
// slot_writes), route_observation.
//
// Carried contracts:
// - The retained expert cache reclaims under the Memory subpackage's
//   budget authority.
// - Previous-token prefetch is demand-led, not speculative-by-default.
// - Expert paging timings feed the switchable performance attribution
//   log.
