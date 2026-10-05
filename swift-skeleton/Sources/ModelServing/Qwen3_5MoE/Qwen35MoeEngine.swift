// Qwen35MoeEngine.swift — ModelServing.Qwen3_5MoE
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5_moe/* (mod.rs
// umbrella) — the Mixture of Experts sibling of the dense qwen3_5 tree.
// Rust keeps the two variants as separate module trees; Swift mirrors that
// split rather than merging them.
//
// Carried contracts:
// - Dense and Mixture of Experts variants keep separate module trees (as in
//   Rust) but derive structural checks (layer count, hidden size, expert
//   group sizes) from config, not duplicated per variant.
// - The MoE engine routes expert memory questions through the Memory
//   subpackage; no family-local memory arithmetic.
