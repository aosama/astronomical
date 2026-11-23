//! Named error constructors for the generation-session lifecycle.
//!
//! Every function here exists so call sites pass a name by path instead of
//! writing a closure (`.ok_or_else(engine_lost_loaded_model_error)`), and the
//! tests lock each message byte-for-byte so the extraction from inline
//! closures stays provably behavior-preserving.

use crate::InferenceEngineError;
use crate::qwen3_5_resident::model::request_errors;

pub(in crate::qwen3_5_resident::inference_execution) fn engine_lost_loaded_model_error()
-> InferenceEngineError {
    super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
}

pub(in crate::qwen3_5_resident::inference_execution) fn missing_image_pad_token_id_error()
-> InferenceEngineError {
    request_errors::invalid_request_error("generation request is missing the image-pad token ID")
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

pub(in crate::qwen3_5_resident::inference_execution) fn eligible_astro_ssd_prompt_cache_capture_has_no_astro_ssd_prompt_cache_owner_error()
-> InferenceEngineError {
    super::fatal_engine_error(
        "eligible astro-ssd-prompt-cache capture has no astro-ssd-prompt-cache owner",
    )
}

pub(in crate::qwen3_5_resident::inference_execution) fn boundary_checkpoint_workspace_bytes_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("boundary checkpoint workspace bytes overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn astro_ssd_prompt_cache_publication_workspace_bytes_overflowed_error()
-> InferenceEngineError {
    super::fatal_engine_error("astro-ssd-prompt-cache publication workspace bytes overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn generation_context_memory_reservation_overflowed_error()
-> InferenceEngineError {
    request_errors::invalid_request_error("generation context memory reservation overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn prefill_activation_workspace_exceeds_platform_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    request_errors::invalid_request_error("prefill activation workspace exceeds the platform range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn generation_context_workspace_reservation_overflowed_error()
-> InferenceEngineError {
    request_errors::invalid_request_error("generation context workspace reservation overflowed")
}

pub(in crate::qwen3_5_resident::inference_execution) fn current_mlx_memory_ceiling_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("current MLX memory ceiling exceeds the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_active_memory_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX active memory exceeds the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_buffer_cache_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX buffer cache memory exceeds the u64 range")
}

pub(in crate::qwen3_5_resident::inference_execution) fn mlx_peak_memory_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    super::fatal_engine_error("MLX peak memory exceeds the u64 range")
}

#[cfg(test)]
#[path = "generation_error_constructors_tests.rs"]
mod generation_error_constructors_tests;
