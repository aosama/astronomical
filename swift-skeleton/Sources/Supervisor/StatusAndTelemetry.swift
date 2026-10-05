// StatusAndTelemetry.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor status and telemetry surfaces
// (instance status, daemon_ipc_models reporting).
//
// Carried contracts:
// - Reported memory figures come from MemoryTelemetry in decimal SI; the
//   supervisor never re-measures or reformats policy figures.
// - Performance logging surfaces stay switchable through configuration.
