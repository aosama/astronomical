//! Decode lifecycle for one resident Qwen3.5 generation session.
//!
//! The children own the answer-generation loop: the one-token-ahead step
//! driver (`advance_generation` — its prefill branch delegates into the
//! prompt-prefill lifecycle, its decode branch samples, applies the
//! repetition penalty, and advances one token), worker-facing token and
//! progress emission, and model-visible feedback injection. This root only
//! wires them into the engine's inference_execution module and re-binds the
//! engine-level owners so the children keep one-level `super::` paths.

pub(in crate::qwen3_5_resident::inference_execution) mod advance_generation;
// One level wider than its sibling: the engine re-exports emission types
// across the whole resident engine, matching the visibility this module had
// before the move.
pub(in crate::qwen3_5_resident::inference_execution) mod decode_step_decisions;
pub(in crate::qwen3_5_resident) mod generated_token_emission;
pub(in crate::qwen3_5_resident::inference_execution) mod inject_input_tokens;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::completed_forward_memory;
use super::engine_request;
use super::fatal_engine_error;
use super::qwen3_5_runtime_error;
