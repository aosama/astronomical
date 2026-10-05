// MultiTokenPrediction.swift — ModelServing.Qwen3_5.MultiTokenPrediction
//
// INERT MIGRATION SKELETON — comments only; nothing in this file compiles.
//
// Migrates from (wave 3): crates/model-serving/src/qwen3_5/
// multi_token_prediction/* (speculative draft heads).
//
// Carried contracts:
// - Draft depth adapts through the Memory subpackage's admission logic
//   (mtp draft depth), never a hardcoded constant.
// - Speculation wins are attributed separately so speedups are measurable,
//   not assumed.
