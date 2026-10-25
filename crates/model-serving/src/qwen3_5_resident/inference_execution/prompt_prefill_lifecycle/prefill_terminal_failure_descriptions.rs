//! Terminal prefill failure descriptions for the resident engine.
//!
//! The resident engine never shrinks a chunk to make a forward fit: every
//! admission shortfall is terminal and demotes the request to the streaming
//! engine with a typed cause. These pure builders construct the exact
//! failure descriptions; their tests lock every template byte-for-byte so
//! the extraction from the inline arms stays provably behavior-preserving.

/// Transient-peak miss: the forward's temporary workspace exceeded the
/// ceiling even though the stable footprint fits.
pub(super) fn transient_peak_shortfall_failure_description(
    attempted_prefill_chunk_token_count: usize,
    admission_reason: &str,
) -> String {
    format!(
        "resident prefill chunk of {attempted_prefill_chunk_token_count} tokens cannot fit its transient peak under the MLX ceiling: {admission_reason}"
    )
}

/// Stable-footprint miss: memory that stays resident after the forward
/// completes does not fit, so no smaller chunk can help.
pub(super) fn stable_footprint_shortfall_failure_description(
    attempted_prefill_chunk_token_count: usize,
    admission_reason: &str,
) -> String {
    format!(
        "attempted prefill chunk of {attempted_prefill_chunk_token_count} tokens cannot fit under the MLX ceiling: {admission_reason}"
    )
}

/// Request-context or runtime-allocation miss: the request's own workspace
/// or an allocation that already failed, neither changed by chunk size.
pub(super) fn request_context_or_allocation_shortfall_failure_description(
    attempted_prefill_chunk_token_count: usize,
    admission_reason: &str,
) -> String {
    format!(
        "attempted prefill chunk of {attempted_prefill_chunk_token_count} tokens cannot proceed on the resident engine: {admission_reason}"
    )
}

/// Active-memory limit breach observed at allocation time inside the forward.
pub(super) fn active_memory_limit_exceeded_failure_description(
    attempted_prefill_chunk_token_count: usize,
    active_memory_bytes: usize,
    attempted_allocation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> String {
    format!(
        "attempted prefill chunk of {attempted_prefill_chunk_token_count} tokens exceeded the MLX active-memory limit: {active_memory_bytes} active bytes, {attempted_allocation_bytes} requested bytes, {allowed_active_memory_bytes} allowed bytes"
    )
}

/// Graphics-processor memory exhaustion raised by the runtime allocator.
pub(super) fn graphics_processor_memory_exhausted_failure_description(
    attempted_prefill_chunk_token_count: usize,
    allocation_reason: &str,
) -> String {
    format!(
        "attempted prefill chunk of {attempted_prefill_chunk_token_count} tokens exhausted GPU memory: {allocation_reason}"
    )
}

/// Byte shortfall of one rejected active-memory allocation, saturating at
/// both the addition and the subtraction.
pub(super) fn active_memory_limit_shortfall_bytes(
    active_memory_bytes: usize,
    attempted_allocation_bytes: usize,
    allowed_active_memory_bytes: usize,
) -> usize {
    active_memory_bytes
        .saturating_add(attempted_allocation_bytes)
        .saturating_sub(allowed_active_memory_bytes)
}

#[cfg(test)]
#[path = "prefill_terminal_failure_descriptions_tests.rs"]
mod prefill_terminal_failure_descriptions_tests;
