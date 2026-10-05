// MemoryReclamation.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory reclamation and
// recovery (eviction ordering, pressure response, recovery paths).
//
// Carried contracts:
// - Reclamation is the sole mutator of residency — no component evicts
//   another component's memory directly.
// - Pressure responses are tested against realistic RAM ratios (about half
//   of model size on disk), not this laptop's limits.
