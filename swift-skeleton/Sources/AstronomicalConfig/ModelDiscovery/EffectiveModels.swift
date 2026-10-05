// EffectiveModels.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/
// effective_models.rs — resolving which classified artifact effectively
// serves each family slot (user selection, defaults, overrides).
//
// Carried contracts:
// - Effective-model resolution happens here, once; downstream crates
//   consume the resolved identity instead of re-scanning directories.
