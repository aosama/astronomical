// Qwen4ExpEngine.swift — ModelServing.Qwen4Exp
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving qwen4_exp family runtime,
// plus crates/config/src/model_discovery/qwen4_exp.rs discovery shapes.
//
// Carried contracts:
// - Structural validity assertions come from config (layer count, hidden
//   size, end tokens) — never golden-master constants.
// - Prefill and generation timings feed the switchable performance
//   attribution log.
