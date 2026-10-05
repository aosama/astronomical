// MemoryResidency.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory residency
// tracking (what is wired, what is streamed, what is evictable).
//
// Carried contracts:
// - Residency classes stay explicit in every decision so reclamation and
//   telemetry reason over the same vocabulary.
// - Residency transitions are attributed in the switchable performance log.
