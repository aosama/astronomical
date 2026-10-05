// ModelFamily.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/
// model_family.rs — the family identity vocabulary shared by discovery and
// serving.
//
// Carried contracts:
// - The family vocabulary lives in config; model-serving dispatches on it
//   but never extends it locally.
