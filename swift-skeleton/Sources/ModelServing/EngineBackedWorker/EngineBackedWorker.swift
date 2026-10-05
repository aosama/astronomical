// EngineBackedWorker.swift — ModelServing.EngineBackedWorker
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/engine_backed_worker/* —
// construction, generation_start, generation_advance, model_swap,
// memory_limit, idle_command, embeddings, image_generation, output,
// protocol, fatal.
//
// Carried contracts:
// - The worker loop is engine-agnostic: family runtimes plug in through
//   the InferenceEngine contract, never by branching on family names here.
// - Model swap and memory-limit handling live in this loop and feed the
//   switchable performance attribution log.
