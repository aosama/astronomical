// Daemon.swift — Supervisor
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/supervisor daemon lifecycle (application
// startup, daemon IPC (Inter-Process Communication) surface, daemon_ipc*
// modules).
//
// Carried contracts:
// - The Stable instance's lifecycle is owned by macOS LaunchAgents; this
//   code never stops or restarts Stable — only Development.
// - Commands emit a live progress indicator instead of silent waits.
