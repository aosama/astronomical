// ArtifactValidation.swift — ModelServing.ArtifactValidation
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/artifact_validation/* —
// bounded_safetensors, raw_safetensors_inventory, required_files,
// safetensors_dtype, shared_blob_resolution, tensor_inventory,
// validated_artifact, validated_safetensors_source.
//
// Carried contracts:
// - Validation asserts structure derived from config (tensor inventory,
//   dtypes, required files, shared-blob resolution) — never golden-master
//   constants like exact byte counts that couple to one packaging variant.
// - Safetensors header and inventory reads stay bounded; a corrupt or
//   truncated artifact is a typed error, not a panic.
