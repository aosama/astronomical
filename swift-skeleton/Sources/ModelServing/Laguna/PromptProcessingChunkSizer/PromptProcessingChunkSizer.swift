// PromptProcessingChunkSizer.swift — ModelServing.Laguna.PromptProcessingChunkSizer
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/laguna/
// prompt_processing_chunk_sizer/* (how large each prefill chunk is).
//
// Carried contracts:
// - Chunk sizing derives from the Memory subpackage's budget arithmetic —
//   it adapts to any laptop, never tuned to one developer machine.
