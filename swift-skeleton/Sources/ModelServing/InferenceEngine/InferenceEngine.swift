// InferenceEngine.swift — ModelServing.InferenceEngine
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/inference_engine/* —
// contract, error, load_result, mlx_owner: the engine surface shared by all
// model families (prompt processing, token generation loop, family-agnostic
// execution contracts).
//
// Carried contracts:
// - Prefill and generation timings feed the switchable performance
//   attribution log (prompt processing and token generation are
//   first-class critical paths).
// - The Romeo and Juliet fixture is the text input for engine tests; no
//   random text or tokens.
