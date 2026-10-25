//! Pure step decisions for one resident Qwen3.5 decode advance.

/// Decides whether reporting a pending one-token-ahead successor should mark
/// the first decode forward as zero milliseconds. Final prefill already
/// produced this token asynchronously, so attributing later output handling
/// to decode would be untruthful — but only the first such successor, and
/// only before any real decode forward has been timed.
pub(in crate::qwen3_5_resident::inference_execution) fn pending_successor_reports_zero_first_decode_time(
    generated_token_count: u16,
    first_decode_forward_elapsed_millis: Option<u64>,
) -> bool {
    generated_token_count == 0 && first_decode_forward_elapsed_millis.is_none()
}

#[cfg(test)]
#[path = "decode_step_decisions_tests.rs"]
mod decode_step_decisions_tests;
