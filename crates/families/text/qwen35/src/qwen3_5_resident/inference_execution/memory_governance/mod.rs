//! Memory governance for one resident Qwen3.5 inference engine.
//!
//! The memory governor owns the limit (`resident_memory_governor`). Before
//! each forward the engine asks it whether the forward fits and afterwards
//! reports what MLX measured (`forward_memory`, `completed_forward_memory`).
//! The request's whole context is checked the same way before the first
//! forward (`initial_context_memory_admission`). `memory_limit` moves the
//! limit, and `request_memory_release` returns MLX's reusable allocations
//! when a request ends.

pub(in crate::qwen3_5_resident::inference_execution) mod completed_forward_memory;
pub(in crate::qwen3_5_resident::inference_execution) mod initial_context_memory_admission;
// One level wider than its siblings: the prompt and reply paths across the
// engine use its forward shape and admission error.
pub(in crate::qwen3_5_resident) mod forward_memory;
pub(in crate::qwen3_5_resident::inference_execution) mod memory_limit;
pub(in crate::qwen3_5_resident::inference_execution) mod resident_memory_governor;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::fatal_runtime_error;
