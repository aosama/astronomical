// SafetensorsSupport.swift — RuntimeIntegration
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/runtime-integration safetensors weight
// reading (header parsing, dtype and shape mapping onto MLX arrays).
//
// Carried contracts:
// - Header parsing stays bounded: untrusted safetensors headers are read
//   with explicit byte caps before any allocation proportional to claimed
//   tensor sizes.
// - Weight-loading timings feed the switchable performance attribution log
//   (model loading from disk is a first-class critical path).
