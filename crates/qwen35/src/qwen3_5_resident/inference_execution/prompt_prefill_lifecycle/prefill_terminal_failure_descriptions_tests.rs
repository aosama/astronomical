use super::{
    active_memory_limit_exceeded_failure_description, active_memory_limit_shortfall_bytes,
    graphics_processor_memory_exhausted_failure_description,
    request_context_or_allocation_shortfall_failure_description,
    stable_footprint_shortfall_failure_description, transient_peak_shortfall_failure_description,
};

#[test]
fn should_describe_a_transient_peak_shortfall_with_the_exact_established_text() {
    let failure_description =
        transient_peak_shortfall_failure_description(2_048, "peak 12 MiB over the allowance");

    assert_eq!(
        failure_description,
        "resident prefill chunk of 2048 tokens cannot fit its transient peak under the MLX ceiling: peak 12 MiB over the allowance"
    );
}

#[test]
fn should_describe_a_stable_footprint_shortfall_with_the_exact_established_text() {
    let failure_description =
        stable_footprint_shortfall_failure_description(561, "stable bytes one over the ceiling");

    assert_eq!(
        failure_description,
        "attempted prefill chunk of 561 tokens cannot fit under the MLX ceiling: stable bytes one over the ceiling"
    );
}

#[test]
fn should_describe_a_request_context_shortfall_with_the_exact_established_text() {
    let failure_description =
        request_context_or_allocation_shortfall_failure_description(1_792, "allocation failed");

    assert_eq!(
        failure_description,
        "attempted prefill chunk of 1792 tokens cannot proceed on the resident engine: allocation failed"
    );
}

#[test]
fn should_describe_an_active_memory_limit_breach_with_the_exact_established_text() {
    let failure_description =
        active_memory_limit_exceeded_failure_description(2_048, 10_485_760, 2_097_152, 12_582_912);

    assert_eq!(
        failure_description,
        "attempted prefill chunk of 2048 tokens exceeded the MLX active-memory limit: 10485760 active bytes, 2097152 requested bytes, 12582912 allowed bytes"
    );
}

#[test]
fn should_describe_graphics_processor_exhaustion_with_the_exact_established_text() {
    let failure_description =
        graphics_processor_memory_exhausted_failure_description(512, "resource unavailable");

    assert_eq!(
        failure_description,
        "attempted prefill chunk of 512 tokens exhausted GPU memory: resource unavailable"
    );
}

#[test]
fn should_compute_the_active_memory_shortfall_bytes_without_underflow() {
    assert_eq!(active_memory_limit_shortfall_bytes(100, 50, 120), 30);
    // Allowed covers everything: no shortfall, never an underflow.
    assert_eq!(active_memory_limit_shortfall_bytes(100, 50, 1_000), 0);
    // Both arithmetic edges saturate in order: the addition saturates first,
    // so the subtraction of one is exact — the original inline arithmetic
    // produced this same value.
    assert_eq!(
        active_memory_limit_shortfall_bytes(usize::MAX, usize::MAX, 1),
        usize::MAX - 1
    );
}
