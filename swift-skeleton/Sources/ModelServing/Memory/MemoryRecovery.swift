// MemoryRecovery.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory/recovery.rs —
// the memory package's recovery path, alongside reclamation.rs and
// phase.rs at the package root.
//
// Carried contracts:
// - Recovery returns the process to a known budgeted state; it never
//   guesses sizes, it re-derives them from the budget.
// - All memory-management code stays in this package — policies,
//   decisions, streaming, and calculations alike.
