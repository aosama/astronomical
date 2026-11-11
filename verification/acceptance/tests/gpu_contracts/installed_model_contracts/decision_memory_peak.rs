//! Reads how much MLX memory a decision engine has needed so far, for the
//! latency journeys of each decision-model family.
//!
//! A catalog entry's `approximate_memory_bytes` tells a user whether a model
//! fits their Mac, so it is recorded from this reading after requests at the
//! model's token limit, not estimated from the weight files. The reading is
//! the allocator's peak since the process started: it covers the load and
//! every request before it, so the value after the last request of a journey
//! is the most the whole journey needed.

use astronomical_model_serving::DecisionEngine;

const BYTES_PER_GIBIBYTE: f64 = 1024.0 * 1024.0 * 1024.0;

/// The peak the engine reported after cleaning up its latest request.
pub(super) fn peak_memory_bytes_after_request(decision_engine: &mut dyn DecisionEngine) -> u64 {
    decision_engine
        .take_post_cleanup_memory_telemetry()
        .expect("a decision engine reports memory telemetry after every request")
        .peak_memory_bytes
}

/// `peak_memory_bytes` as the fields a journey prints beside its timings.
pub(super) fn peak_memory_report(peak_memory_bytes: u64) -> String {
    format!(
        "peak_memory_bytes={peak_memory_bytes} peak_memory_gibibytes={:.2}",
        peak_memory_bytes as f64 / BYTES_PER_GIBIBYTE
    )
}
