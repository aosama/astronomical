// PersistentCache.swift — ModelServing.PersistentCache
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/persistent_cache/* —
// block format and keys, disk store with global quota, retention policy,
// visual embedding contracts (on-disk KV (key-value) and artifact cache
// layers).
//
// Carried contracts:
// - SSD (Solid State Drive) streaming tests allocate RAM at roughly half the
//   model's on-disk size, mirroring realistic end-user machines.
// - Real-model SSD streaming journeys measure tokens per second for both
//   prefill and generation.
// - Cache timings feed the switchable performance attribution log (disk
//   cache is a first-class critical path).
