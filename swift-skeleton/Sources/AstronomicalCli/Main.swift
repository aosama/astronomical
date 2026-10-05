// Main.swift — AstronomicalCli
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/astronomical CLI (Command Line Interface)
// entry point and command registry.
//
// Carried contracts:
// - Commands emit a live progress indicator instead of silent waits.
// - The CLI resolves default models through the shared AstronomicalConfig
//   default-model surface so CLI and daemon can never drift.
