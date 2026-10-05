// DeepseekV4Engine.swift — ModelServing.DeepseekV4
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving deepseek_v4 family runtime,
// plus crates/config/src/model_discovery/deepseek_v4.rs discovery shapes.
//
// Carried contracts:
// - Structural validity assertions come from config — never golden-master
//   constants coupled to one packaging variant.
// - Sparse-expert execution goes through SparseExperts; this module owns
//   family structure, not paging policy.
