// ModelDiscovery.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/* plus
// crates/config/src/model_discovery_huggingface_cache.rs — artifact
// discovery, bounded artifact reads, classified artifacts, effective models,
// and the per-family discovery modules (qwen3_5, qwen4_exp, qwen_image_21
// with its documents module, flux2_klein with its documents module, laguna,
// k2_horizon_mova, deepseek_v4, modernbert, model_family).
//
// Carried contracts:
// - Every artifact read stays bounded by explicit byte caps per artifact
//   type (family config 4 MiB, pipeline index 1 MiB, and the rest).
// - Directory scan depth stays capped (currently 4).
// - Hugging Face cache entries resolve by leaf repo id: "models--org--name"
//   yields "name".
// - Discovered identities derive revisions from a hash of config bytes, so
//   identity changes exactly when the artifact changes.
