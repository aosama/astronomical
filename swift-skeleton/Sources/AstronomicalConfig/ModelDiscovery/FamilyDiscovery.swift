// FamilyDiscovery.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): the per-family discovery shapes in
// crates/config/src/model_discovery/ — deepseek_v4.rs, flux2_klein.rs,
// flux2_klein_documents.rs, k2_horizon_mova.rs, laguna.rs, modernbert.rs,
// qwen3_5.rs, qwen4_exp.rs, qwen_image_21.rs,
// qwen_image_21_documents.rs.
//
// Carried contracts:
// - Each family declares only the artifact roles it actually ships; no
//   shared mega-schema with family-irrelevant fields.
// - The two documents modules (flux2_klein, qwen_image_21) keep their own
//   bounded-read rules for text-encoder document bundles.
