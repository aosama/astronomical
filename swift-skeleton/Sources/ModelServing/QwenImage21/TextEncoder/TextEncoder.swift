// TextEncoder.swift — ModelServing.QwenImage21.TextEncoder
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen_image_21/
// text_encoder/* (prompt encoding; the family root keeps its flat
// text_conditioning beside it, as in Rust).
//
// Carried contracts:
// - Text-conditioning timings feed the switchable performance attribution
//   log; they are part of the serving critical path.
