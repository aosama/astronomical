// RuntimeConfig.swift — AstronomicalConfig
//
// MIGRATION MARKER — comment-only placeholder; the package manifest is real
// and this module builds, but the unit below is not ported yet.
//
// Migrates from (wave 1): crates/config/src/{config_document,config_file,
// chunking_config,logging_config,prompt_cache_config,maximum_mlx_memory,
// resolved_model_config,legacy_config_migration,duplicate_key_json,
// configuration_generation,config_error}.rs
//
// Ported so far into sibling files in this module:
// - config_document + config_file (config load / strict parse / atomic
//   first-run write): AstronomicalConfig.swift, ConfigFileStore.swift,
//   StrictJson.swift, UserConfigFile.swift, RuntimeConfigFile.swift,
//   PromptCacheConfigFile.swift, ChunkingConfigFile.swift,
//   DiagnosticsConfigFile.swift, ModelConfigFile.swift,
//   ModelLimitsConfigFile.swift, GenerationDefaultsConfigFile.swift.
//
// Still to port (each with its own slice):
// - chunking_config resolution and persist-back
// - logging_config
// - maximum_mlx_memory
// - resolved_model_config (per-model limits, generation-defaults range
//   validation, model-ID hygiene)
// - legacy_config_migration
// - duplicate_key_json
// - configuration_generation
// - config_error (remaining error cases beyond the load journey)
//
// Carried contracts:
// - Bounded reads stay bounded: user config files cap at 1 MiB.
// - Duplicate keys in JSON stay a hard rejection, not a last-wins silently.
// - Config writes stay atomic byte transactions with the schema written
//   adjacent to the file.
// - User-facing sizes stay decimal SI (1 GB = 1,000,000,000 bytes).
