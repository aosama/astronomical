// ModelFamilyRuntime.swift — ModelServing.ModelFamilyRuntime
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/model_family_runtime/* —
// inference_engine, image_engine, request, output, processor: the
// family-runtime registry and dispatch (how a discovered family identity
// selects its engine implementation).
//
// Carried contracts:
// - Family selection derives from the config crate's discovery identities;
//   the registry never re-parses artifacts to guess a family.
// - Structural validity checks come from config (layer count, hidden size,
//   shard count, affine profiles, end tokens) — never golden-master
//   constants coupled to one packaging variant.
