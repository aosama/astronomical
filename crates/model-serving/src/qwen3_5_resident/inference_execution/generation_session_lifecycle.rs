//! Generation-session lifecycle for one resident Qwen3.5 request.
//!
//! The children own the ordered stages — validate, admit, restore, install,
//! and finalize — as independently unit-testable units. This root only wires
//! them into the engine's inference_execution module and re-binds the
//! engine-level owners below so the children keep one-level `super::` paths.

mod generation_error_constructors;
mod generation_finalization;
mod generation_request_validation;
mod generation_restore_position_derivation;
mod generation_session_decisions;
mod generation_session_validate_admit_restore_install;
mod generation_thinking_end_markers;
mod generation_visual_embedding_validation;
mod generation_vocabulary_boundaries;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::decoder_state_reuse;
use super::engine_request;
use super::fatal_engine_error;
use super::persistent_prompt_cache_visual_identity;
use super::qwen3_5_runtime_error;
use super::retained_live_prefix;
use super::visual_embeddings_runtime;
