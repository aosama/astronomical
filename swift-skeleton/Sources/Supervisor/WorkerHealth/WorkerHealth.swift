// WorkerHealth.swift — Supervisor.WorkerHealth
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor/src/worker_health/* —
// worker_health.rs at the crate root plus its publisher.rs submodule.
//
// Carried contracts:
// - Health publication is push-based through the publisher; the supervisor
//   never polls worker liveness by inference from silence alone.
// - Health state drives worker replacement decisions; both stay in the
//   supervisor, never in the worker itself.
