// InferenceExecution.swift — ModelServing.Qwen3_5.InferenceExecution
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5/
// inference_execution/* including its prompt_processing_chunk_sizer
// submodule (prefill and decode orchestration for the family).
//
// Carried contracts:
// - Prompt processing and token generation both feed the switchable
//   performance attribution log.
// - Chunk sizing derives from the Memory subpackage's budget arithmetic.
