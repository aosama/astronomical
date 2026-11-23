use super::*;

// Positive: a refused request carries its reason as an invalid request.
#[test]
fn should_carry_the_reason_of_an_invalid_request() {
    let engine_error = invalid_request_error("the prompt is empty");

    assert!(matches!(
        engine_error,
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "the prompt is empty"
    ));
}

// Negative: the request-shape overflows are the caller's to fix, so they are
// invalid requests, never engine failures.
#[test]
fn should_report_request_overflows_as_invalid_requests() {
    assert!(matches!(
        thinking_budget_output_reservation_overflow_error(),
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "thinking-budget output reservation overflowed"
    ));
    assert!(matches!(
        generation_context_token_count_overflow_error(),
        crate::InferenceEngineError::InvalidRequest { reason } if reason == "generation context token count overflowed"
    ));
}

// Edge: a counter or memory figure past its range is an engine failure with
// the exact text both engines report.
#[test]
fn should_carry_the_exact_range_failure_texts() {
    let try_from_int_error = u32::try_from(usize::MAX).expect_err("usize maximum must exceed u32");

    assert!(matches!(
        generated_token_counter_overflowed_error(),
        crate::InferenceEngineError::Fatal { reason } if reason == "generated-token counter overflowed"
    ));
    assert!(matches!(
        mlx_active_memory_bytes_exceed_u64_range_error(try_from_int_error),
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX active memory bytes exceed the u64 range"
    ));
    assert!(matches!(
        mlx_buffer_cache_bytes_exceed_u64_range_error(try_from_int_error),
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX buffer cache memory bytes exceed the u64 range"
    ));
    assert!(matches!(
        mlx_peak_memory_bytes_exceed_u64_range_error(try_from_int_error),
        crate::InferenceEngineError::Fatal { reason } if reason == "MLX peak memory bytes exceed the u64 range"
    ));
}
