//! What every paired policy run, and every pair, must satisfy before its numbers
//! enter the comparison.

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;

use super::MeasuredShapeWindow;
use crate::performance_throughput::completion::ThroughputMeasurement;
use crate::performance_throughput::support::{self as throughput_support, ThroughputJourney};

pub(super) fn assert_measured_policy_run(
    journey: &ThroughputJourney,
    measured_shape_window: &MeasuredShapeWindow,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    measured_throughput: &ThroughputMeasurement,
) {
    measured_shape_window.assert_measurement_fits(measured_throughput);
    assert_eq!(
        measured_throughput.decode_astro_expert_cache_policy, decode_astro_expert_cache_policy,
        "the worker measurement must retain the selected decode astro expert cache policy"
    );
    assert!(
        measured_throughput.measured_output_text.is_some(),
        "the completed measurement must preserve streamed output for parity checks"
    );
    assert_peak_memory_within_limit(journey, measured_throughput);
    if journey.attribution_enabled {
        let diagnostic_evidence = measured_throughput
            .decode_astro_expert_cache_diagnostic_evidence
            .as_ref()
            .expect("the attributed run must produce measured decode astro expert cache evidence");
        assert!(
            diagnostic_evidence
                .decode_token_forward_latency_p95_nanoseconds
                .is_some(),
            "the generation report must include decode-forward p95"
        );
        assert!(
            diagnostic_evidence
                .decode_token_forward_latency_sample_count
                .is_some_and(|sample_count| sample_count > 0),
            "the generation report must include decode-forward histogram samples"
        );
        assert_eq!(
            diagnostic_evidence.decode_expert_source_payload_bytes,
            diagnostic_evidence.decode_expert_source_summary_payload_bytes,
            "the decode source counter must agree with its phase summaries"
        );
    }
}

/// The two policies of one pair must write the same text, and when the run is
/// attributed they must read the same prefill source bytes.
pub(super) fn assert_pair_parity(
    baseline: &ThroughputMeasurement,
    candidate: &ThroughputMeasurement,
    attribution_enabled: bool,
) {
    assert_eq!(
        baseline.measured_output_text, candidate.measured_output_text,
        "baseline and candidate must generate identical Romeo and Juliet output"
    );
    if attribution_enabled {
        assert_eq!(
            baseline
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .expect("the baseline attribution must be present")
                .prefill_expert_source_payload_bytes,
            candidate
                .decode_astro_expert_cache_diagnostic_evidence
                .as_ref()
                .expect("the candidate attribution must be present")
                .prefill_expert_source_payload_bytes,
            "prefill source bytes must remain identical across decode astro expert cache policies"
        );
    }
}

fn assert_peak_memory_within_limit(
    journey: &ThroughputJourney,
    measured_throughput: &ThroughputMeasurement,
) {
    let diagnostic_peak_memory_bytes = measured_throughput
        .decode_astro_expert_cache_diagnostic_evidence
        .as_ref()
        .and_then(|diagnostic_evidence| diagnostic_evidence.mlx_peak_memory_bytes);
    let observed_peak_memory_bytes = [
        measured_throughput.mlx_peak_memory_bytes,
        diagnostic_peak_memory_bytes,
    ]
    .into_iter()
    .flatten()
    .max()
    .expect("the worker must report a measured peak memory value");
    let maximum_mlx_memory_bytes = journey
        .maximum_mlx_memory_bytes
        .unwrap_or(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    assert!(
        observed_peak_memory_bytes <= maximum_mlx_memory_bytes,
        "the {observed_peak_memory_bytes}-byte observed peak exceeded the {maximum_mlx_memory_bytes}-byte configured memory ceiling"
    );
}
