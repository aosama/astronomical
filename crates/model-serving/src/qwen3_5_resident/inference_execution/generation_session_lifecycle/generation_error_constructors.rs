//! Named error constructors for the generation-session lifecycle.
//!
//! Every function here exists so call sites pass a name by path instead of
//! writing a closure (`.ok_or_else(engine_lost_loaded_model_error)`), and the
//! tests lock each message byte-for-byte so the extraction from inline
//! closures stays provably behavior-preserving.

use crate::InferenceEngineError;
use crate::qwen3_5_resident::model::memory_admission;

pub(super) fn engine_lost_loaded_model_error() -> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
}

pub(super) fn missing_image_pad_token_id_error() -> InferenceEngineError {
    memory_admission::invalid_request_error("generation request is missing the image-pad token ID")
}

pub(super) fn visual_embedding_model_contract_not_loaded_error() -> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 persistent visual embedding model contract is not loaded")
}

pub(super) fn thinking_budget_output_reservation_overflow_error() -> InferenceEngineError {
    memory_admission::invalid_request_error("thinking-budget output reservation overflowed")
}

pub(super) fn generation_context_token_count_overflow_error() -> InferenceEngineError {
    memory_admission::invalid_request_error("generation context token count overflowed")
}

pub(super) fn mlx_active_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX active memory bytes exceed the u64 range")
}

pub(super) fn mlx_allocator_cache_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX allocator-cache memory bytes exceed the u64 range")
}

pub(super) fn mlx_peak_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX peak memory bytes exceed the u64 range")
}

pub(super) fn visual_embedding_row_count_exceeds_i32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("visual embedding row count exceeds the i32 range")
}

pub(super) fn visual_embedding_hidden_size_exceeds_i32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("visual embedding hidden size exceeds the i32 range")
}

pub(super) fn restored_prompt_prefix_exceeds_u32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("restored prompt prefix exceeds the u32 range")
}

#[cfg(test)]
#[path = "generation_error_constructors_tests.rs"]
mod generation_error_constructors_tests;
