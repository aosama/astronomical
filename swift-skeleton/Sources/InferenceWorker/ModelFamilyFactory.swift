// ModelFamilyFactory.swift — InferenceWorker
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): apps/inference-worker model_family_factory —
// mapping a worker model configuration onto a ModelServing family runtime.
//
// Carried contracts:
// - Family selection derives from AstronomicalConfig discovery identities;
//   the factory never guesses families from filenames.
// - Model-load timings feed the switchable performance attribution log
//   (loading from disk is a first-class critical path).
// - Tests exercise the factory with the Romeo and Juliet fixture inputs,
//   never random text or tokens.
