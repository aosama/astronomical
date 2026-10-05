// MemoryTelemetry.swift — ModelServing.Memory
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/memory telemetry
// (observed usage and decision traces surfaced to status and logging).
//
// Carried contracts:
// - Telemetry reports the same byte figures the policies decided on — no
//   re-measurement drift between decision and report.
// - User-facing figures use decimal SI (1 GB = 1,000,000,000 bytes).
