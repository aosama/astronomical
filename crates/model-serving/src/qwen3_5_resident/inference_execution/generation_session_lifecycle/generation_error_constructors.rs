//! Named error constructors for the generation-session lifecycle.
//!
//! Every function here exists so call sites pass a name by path instead of
//! writing a closure (`.ok_or_else(engine_lost_loaded_model_error)`), and the
//! tests lock each message byte-for-byte so the extraction from inline
//! closures stays provably behavior-preserving.

use crate::InferenceEngineError;
use crate::qwen3_5_resident::model::memory_admission;

pub(in crate::qwen3_5_resident::inference_execution) fn engine_lost_loaded_model_error()
-> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
}

pub(in crate::qwen3_5_resident::inference_execution) fn missing_image_pad_token_id_error()
-> InferenceEngineError {
    memory_admission::invalid_request_error("generation request is missing the image-pad token ID")
}

pub(in crate::qwen3_5_resident::inference_execution) fn visual_embedding_model_contract_not_loaded_error()
-> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 persistent visual embedding model contract is not loaded")
}

pub(in crate::qwen3_5_resident::inference_execution) fn thinking_budget_output_reservation_overflow_error()
-> InferenceEngineError {
    memory_admission::invalid_request_error("thinking-budget output reservation overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn generation_context_token_count_overflow_error()
-> InferenceEngineError {
    memory_admission::invalid_request_error("generation context token count overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_active_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX active memory bytes exceed the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_allocator_cache_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX allocator-cache memory bytes exceed the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_peak_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX peak memory bytes exceed the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn visual_embedding_row_count_exceeds_i32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("visual embedding row count exceeds the i32 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn visual_embedding_hidden_size_exceeds_i32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("visual embedding hidden size exceeds the i32 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn restored_prompt_prefix_exceeds_u32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("restored prompt prefix exceeds the u32 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn completed_prefill_chunk_tokens_exceed_u32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("completed_prefill_chunk_tokens exceeds the u32 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn eligible_prompt_cache_capture_has_no_persistent_cache_owner_error()
-> InferenceEngineError {
    super::fatal_engine_error("eligible prompt-cache capture has no persistent cache owner")
}

pub(in crate::qwen3_5_resident::inference_execution) fn boundary_checkpoint_workspace_bytes_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("boundary checkpoint workspace bytes overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn prompt_cache_publication_workspace_bytes_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("prompt-cache publication workspace bytes overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn generated_token_counter_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("generated-token counter overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn full_attention_growth_tokens_exceed_i32_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("full-attention growth tokens exceed Int32 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn context_memory_reservation_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 context memory reservation overflowed")
}

#[cfg(test)]
#[path = "generation_error_constructors_tests.rs"]
mod generation_error_constructors_tests;
