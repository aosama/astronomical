// ModelsCommand.swift — AstronomicalCli
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/astronomical models verb — listing
// discovered and catalog models, setting the default model.
//
// Carried contracts:
// - Listed identities come from AstronomicalConfig discovery and the
//   supervisor's library catalog; the CLI keeps no model table of its own.
// - User-facing sizes stay decimal SI (1 GB = 1,000,000,000 bytes).
