// LagunaArtifacts.swift — ModelServing.Laguna.Artifacts
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/artifacts/*
// (family artifact loading and validation).
//
// Carried contracts:
// - Bounded artifact reads stay bounded (index 32 MiB, text document
//   32 MiB).
// - Structural validity comes from config, never golden-master constants.
