// Qwen35TextEngine.swift — ModelServing.Qwen3_5
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5 (dense) family
// runtime, plus crates/config/src/model_discovery/qwen3_5.rs discovery
// shapes. The Mixture of Experts variant is a separate Rust module tree
// (qwen3_5_moe) and gets its own Swift module folder: Qwen3_5MoE.
//
// Carried contracts:
// - Dense and Mixture of Experts variants keep separate module trees (as in
//   Rust) but derive structural checks (layer count, hidden size, expert
//   group sizes) from config, not duplicated per variant.
// - Structural validity assertions come from config — never golden-master
//   constants coupled to one quantization artifact.
