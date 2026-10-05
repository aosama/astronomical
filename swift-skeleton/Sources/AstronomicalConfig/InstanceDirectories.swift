// InstanceDirectories.swift — AstronomicalConfig
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/astronomical_runtime_instance.rs.
//
// Carried contracts:
// - Stable and Development instances stay distinct: separate state
//   directories, separate ports, separate app bundles. The Stable instance
//   is the user's daily driver and is never restarted by code or tests.
// - Instance locations resolve through platform-standard application
//   directories and configuration — never hardcoded developer paths.
