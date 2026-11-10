//! Persistent prompt-cache storage and reuse for one resident Qwen3.5 engine.
//!
//! The children own one concern: durable prompt-prefix reuse. The restore
//! path (`decoder_state_reuse`) rebuilds decoder state from on-disk blocks,
//! capture publishes new blocks after durable boundaries through the shared
//! `qwen3_5_core::prompt_cache_capture`, with this engine's retry behaviour
//! in `prompt_cache_capture_engine`, identity (`visual_identity`) decides
//! block eligibility, in-memory adoption (`retained_live_prefix`, adopt)
//! reuses the still-live previous state, and startup logging reports the
//! store's condition. Both phase folders consume this machinery; this root
//! only wires it into the engine's inference_execution module and re-binds
//! the engine-level owners so the children keep one-level `super::` paths.

pub(in crate::qwen3_5_resident::inference_execution) mod adopt_retained_live_prefix;
pub(in crate::qwen3_5_resident::inference_execution) mod decoder_state_reuse;
pub(in crate::qwen3_5_resident::inference_execution) mod in_memory_adoption_decision;
pub(in crate::qwen3_5_resident::inference_execution) mod persistent_prompt_cache_startup_logging;
pub(in crate::qwen3_5_resident::inference_execution) mod persistent_prompt_cache_visual_identity;
mod prompt_cache_capture_engine;
pub(in crate::qwen3_5_resident::inference_execution) mod restore_transient_workspace;
// One level wider than its siblings: the engine module re-exports the
// retained-prefix owner across the whole resident engine, matching the
// visibility this module had before the move.
pub(in crate::qwen3_5_resident) mod retained_live_prefix;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::engine_request;
use super::fatal_engine_error;
use super::qwen3_5_runtime_error;
