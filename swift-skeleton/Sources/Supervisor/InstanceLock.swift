// InstanceLock.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor single-instance lock and state
// directory ownership.
//
// Carried contracts:
// - Stable and Development instances stay isolated: separate state
//   directories and ports; a Development run can never take Stable's lock.
// - Paths resolve through InstanceDirectories, never hardcoded developer
//   locations.
