// StructuredGeneration.swift — ModelServing.StructuredGeneration
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/structured_generation/* —
// json_prefix, regex_mask (grammar and schema-constrained decoding).
//
// Carried contracts:
// - Grammar and schema inputs are untrusted: parsing stays bounded and
//   produces typed errors, never panics.
// - Constraint-application overhead is attributed in the switchable
//   performance log so it cannot hide generation slowdowns.
