// MemoryBudget.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory budget
// calculation (wired-memory limits, model-size-to-budget arithmetic).
//
// Carried contracts:
// - Budgets adapt to any laptop, any RAM size, any GPU (Graphics Processing
//   Unit) wired memory limit — never tuned to one developer machine.
// - Byte arithmetic stays in plain bytes internally; user-facing reporting
//   converts to decimal SI (1 GB = 1,000,000,000 bytes).
