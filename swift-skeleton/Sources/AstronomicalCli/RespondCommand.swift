// RespondCommand.swift — AstronomicalCli
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): apps/astronomical respond verb (and embed) —
// one-shot generation and embedding requests against the daemon.
//
// Carried contracts:
// - Temperature is never forced to 0; the provider decides.
// - One-shot requests reuse the daemon's effective-default resolution
//   rather than a CLI-local default model.
