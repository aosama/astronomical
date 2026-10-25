//! Memory governance for one resident Qwen3.5 inference engine.
//!
//! The children own every byte the engine spends, as one budget loop:
//! the ceiling policy (`memory_limit`), the session-start context gate
//! (`initial_context_memory_admission`), per-forward fixed-shape admission
//! (`memory_admission`), post-forward measurement that teaches future
//! admission (`completed_forward_memory`), and final synchronized allocator
//! release (`request_memory_release`). Admission predicts before a forward
//! begins; observation corrects the predictions after it ends. This root
//! only wires them into the engine's inference_execution module and
//! re-binds the engine-level owners so the children keep one-level
//! `super::` paths.

pub(in crate::qwen3_5_resident::inference_execution) mod completed_forward_memory;
pub(in crate::qwen3_5_resident::inference_execution) mod initial_context_memory_admission;
// One level wider than its siblings: model-level helpers across the whole
// resident engine reference this module's typed admission surface, matching
// the visibility it had before the move.
pub(in crate::qwen3_5_resident) mod memory_admission;
pub(in crate::qwen3_5_resident::inference_execution) mod memory_limit;
pub(in crate::qwen3_5_resident::inference_execution) mod request_memory_release;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::fatal_engine_error;
use super::qwen3_5_runtime_error;
