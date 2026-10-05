// ArtifactDiscovery.swift — AstronomicalConfig.ModelDiscovery
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 1): crates/config/src/model_discovery/
// artifact_discovery.rs — directory scanning for family artifacts.
//
// Carried contracts:
// - Directory scan depth stays capped (currently 4).
// - Discovery reads stay bounded by explicit byte caps per artifact type;
//   scanning never degenerates into unbounded filesystem walks.
