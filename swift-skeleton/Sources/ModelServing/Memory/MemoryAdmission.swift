// MemoryAdmission.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory admission
// decisions.
//
// Carried contracts:
// - All memory-management code lives in this Memory subpackage — policies,
//   decisions, streaming, and calculations. Nothing memory-related lives
//   elsewhere.
// - Admission decisions stay specific: exact byte budgets and residency
//   class, never a vague pass/fail.
