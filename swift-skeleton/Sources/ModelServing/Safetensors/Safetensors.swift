// Safetensors.swift — ModelServing.Safetensors
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/safetensors/* —
// header.rs (model-serving's own safetensors header parsing). Distinct
// from RuntimeIntegration's MLX safetensors I/O support — Rust keeps both,
// so Swift keeps both.
//
// Carried contracts:
// - Header parsing stays bounded; malformed JSON (JavaScript Object
//   Notation) headers produce typed errors, never panics.
