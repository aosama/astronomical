// GenerationQueue.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor queue — request admission and
// ordering across workers.
//
// Carried contracts:
// - Queue-depth limits keep producing HTTP 429 (Too Many Requests) under
//   pressure; the limit derivation moves with the queue, not into endpoint
//   handlers.
// - Queue wait times feed the switchable performance attribution log so
//   admission stalls are attributable.
