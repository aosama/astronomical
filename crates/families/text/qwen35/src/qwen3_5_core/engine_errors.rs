//! Error constructors shared by both Qwen3.5 inference engines, so the same
//! failure reads the same whichever engine reports it.

use crate::InferenceEngineError;

pub fn qwen3_5_runtime_error(runtime_error: impl std::fmt::Display) -> InferenceEngineError {
    fatal_engine_error(runtime_error.to_string())
}

pub fn fatal_engine_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::Fatal {
        reason: reason.into(),
    }
}

/// A request the model cannot serve as asked; the client changes the request.
pub fn invalid_request_error(reason: impl Into<String>) -> InferenceEngineError {
    InferenceEngineError::InvalidRequest {
        reason: reason.into(),
    }
}

/// A reply longer than the token counter can count; the engine stops it.
pub fn generated_token_counter_overflowed_error() -> InferenceEngineError {
    fatal_engine_error("generated-token counter overflowed")
}

/// MLX reported more active memory than a u64 can hold, which only a
/// corrupted reading can produce.
pub fn mlx_active_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    fatal_engine_error("MLX active memory bytes exceed the u64 range")
}

/// The same for the `mlx_buffer_cache` figure.
pub fn mlx_buffer_cache_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    fatal_engine_error("MLX buffer cache memory bytes exceed the u64 range")
}

/// The same for the peak memory figure.
pub fn mlx_peak_memory_bytes_exceed_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    fatal_engine_error("MLX peak memory bytes exceed the u64 range")
}

/// A thinking allowance so large its output reservation overflows; the
/// client asked for an impossible budget.
pub fn thinking_budget_output_reservation_overflow_error() -> InferenceEngineError {
    invalid_request_error("thinking-budget output reservation overflowed")
}

/// A prompt plus output budget past the integer range; the client asked for
/// an impossible context.
pub fn generation_context_token_count_overflow_error() -> InferenceEngineError {
    invalid_request_error("generation context token count overflowed")
}

#[cfg(test)]
#[path = "engine_errors_tests.rs"]
mod engine_errors_tests;

/// The installed MLX ceiling does not fit a u64, which only a corrupted
/// limit can produce on a 64-bit Mac.
pub fn mlx_memory_ceiling_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    fatal_engine_error("current MLX memory ceiling exceeds the u64 range")
}

/// The same for the installed `mlx_buffer_cache` limit.
pub fn mlx_buffer_cache_limit_exceeds_u64_range_error(
    _try_from_error: std::num::TryFromIntError,
) -> InferenceEngineError {
    fatal_engine_error("updated MLX buffer cache limit exceeds the u64 range")
}

/// A ceiling change arrived before any model was loaded to move.
pub fn mlx_memory_ceiling_change_before_load_error() -> InferenceEngineError {
    fatal_engine_error("cannot update the MLX memory ceiling before the model is loaded")
}
