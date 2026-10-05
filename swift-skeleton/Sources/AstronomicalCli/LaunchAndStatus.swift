// LaunchAndStatus.swift — AstronomicalCli
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/astronomical launch, status, schema, and
// validate verbs.
//
// Carried contracts:
// - Status output reads supervisor telemetry as-is; no CLI-side
//   re-measurement of memory or throughput figures.
// - The CLI only ever targets the Development instance during development
//   work; Stable's lifecycle is untouchable.
