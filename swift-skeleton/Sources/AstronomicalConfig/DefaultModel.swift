// DefaultModel.swift — AstronomicalConfig
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/default_model.rs and
// crates/config/src/model_identity.rs.
//
// Carried contracts:
// - The built-in default model id stays shared by the daemon's
//   effective-default resolution and the CLI's request resolution so the two
//   can never drift.
// - Default-model mutations stay exact byte transactions: prior bytes and
//   candidate bytes, applied atomically, never in-place edits.
// - Legacy config migration keeps preserving a backup of the prior file.
