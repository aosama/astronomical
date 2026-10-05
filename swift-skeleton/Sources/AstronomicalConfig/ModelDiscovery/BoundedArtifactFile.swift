// BoundedArtifactFile.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/
// bounded_artifact_file.rs — size-capped reads of artifact files.
//
// Carried contracts:
// - Every artifact read stays bounded by explicit byte caps per artifact
//   type (family config 4 MiB, pipeline index 1 MiB, and the rest).
// - Caps are declared per artifact type, never per machine or developer
//   workstation.
