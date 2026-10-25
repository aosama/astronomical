//! Prompt-prefill lifecycle for one resident Qwen3.5 generation session.
//!
//! The children own the chunk loop in its documented correctness order —
//! plan the chunk, admit it under adaptive memory, dispatch the forward,
//! publish durable prompt-cache state, then advance the cursor — with the
//! terminal seed and per-chunk observability alongside. This root only wires
//! them into the engine's inference_execution module and re-binds the
//! engine-level owners below so the children keep one-level `super::` paths.

mod prefill_advance;
mod prefill_chunk_planning;
// Visible one level up: the engine module re-exports these phase-owned types
// that callers outside the prefill phase consume.
pub(super) mod prefill_execution_context;
mod prefill_forward_dispatch;
mod prefill_invalid_input_error_constructors;
mod prefill_memory_admission;
mod prefill_terminal_failure_descriptions;
mod prompt_prefill;
mod prompt_prefill_counters;
mod prompt_prefill_errors;
pub(super) mod prompt_processing_chunk_sizer;
mod terminal_prefill_seed;

// Re-bindings of the engine-level owners that stay in inference_execution.
// They are private here, but every child above can reach them as
// `super::<name>` exactly as before the move.
use super::Qwen3_5EngineState;
use super::completed_forward_memory;
use super::engine_request;
use super::fatal_engine_error;
use super::memory_admission;
use super::persistent_prompt_cache_capture;
use super::qwen3_5_runtime_error;
