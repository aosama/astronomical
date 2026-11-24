//! Decode lifecycle for one resident Qwen3.5 generation session.
//!
//! The children hold the resident engine's part of the answer-generation
//! loop: its decode-step hooks (`advance_generation`; the step itself is
//! shared in `qwen3_5_core::shared_execution::decode_step`) and model-visible
//! feedback injection. This root only wires them into the engine's
//! inference_execution module and re-binds the engine-level owners so the
//! children keep one-level `super::` paths.

pub(in crate::qwen3_5_resident::inference_execution) mod advance_generation;
pub(in crate::qwen3_5_resident::inference_execution) mod inject_input_tokens;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::completed_forward_memory;
use super::engine_request;
use super::forward_memory;
