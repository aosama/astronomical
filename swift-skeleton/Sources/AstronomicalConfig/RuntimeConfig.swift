// RuntimeConfig.swift — AstronomicalConfig
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/{config_document,config_file,
// chunking_config,logging_config,prompt_cache_config,maximum_mlx_memory,
// resolved_model_config,legacy_config_migration,duplicate_key_json,
// configuration_generation,config_error}.rs
//
// Carried contracts:
// - Bounded reads stay bounded: user config files cap at 1 MiB.
// - Duplicate keys in JSON stay a hard rejection, not a last-wins silently.
// - Config writes stay atomic byte transactions with the schema written
//   adjacent to the file.
// - User-facing sizes stay decimal SI (1 GB = 1,000,000,000 bytes).
