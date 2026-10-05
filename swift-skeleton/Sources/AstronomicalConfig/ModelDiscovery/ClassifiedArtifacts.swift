// ClassifiedArtifacts.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/
// classified_artifacts.rs — the result of classifying discovered files
// into family-specific artifact roles.
//
// Carried contracts:
// - Classification decides family and variant from config-derived
//   structure; it never loads model weights to guess.
// - Discovered identities derive revisions from a hash of config bytes, so
//   identity changes exactly when the artifact changes.
