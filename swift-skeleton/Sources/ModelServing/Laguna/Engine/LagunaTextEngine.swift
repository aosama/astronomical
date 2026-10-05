// LagunaTextEngine.swift — ModelServing.Laguna.Engine
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/engine/* (text
// family runtime), plus crates/config/src/model_discovery/laguna.rs and the
// laguna template source in crates/config.
//
// Carried contracts:
// - Bounded artifact reads stay bounded (index 32 MiB, text document
//   32 MiB, template 512 KiB).
// - Template resolution keeps its dedicated bounded path rather than
//   growing ad-hoc file reads.
