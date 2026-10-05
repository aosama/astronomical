// ModelsAndErrors.swift — RestContract
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 2): crates/rest-contract models-listing shapes and the
// shared error/failure-code envelope.
//
// Carried contracts:
// - The four failure codes live here once and nowhere else; every endpoint
//  composes them instead of inventing per-endpoint errors.
// - Model listing entries derive from the same discovery identities as the
//   config crate, so listed ids and loadable ids cannot drift.
