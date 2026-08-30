use serde::{Deserialize, Serialize};

/// One independently timed fixed-depth or target-only acceptance cell.
///
/// The persisted document intentionally contains only locally measured evidence. Publisher claims,
/// model-card speed numbers, and machine paths never participate in the release decision.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct MtpDepthMeasurement {
    pub(crate) cell_name: String,
    pub(crate) draft_depth: Option<u8>,
    pub(crate) output_token_count: usize,
    pub(crate) generation_elapsed_seconds: f64,
    pub(crate) total_request_elapsed_seconds: f64,
    pub(crate) tokens_per_second: f64,
    pub(crate) maximum_active_mlx_memory_bytes: u64,
    pub(crate) maximum_peak_mlx_memory_bytes: u64,
    pub(crate) mlx_memory_ceiling_bytes: u64,
    pub(crate) operational_fallback_count: u64,
    pub(crate) proposed_draft_count: u64,
    pub(crate) effective_depth_total: u64,
    pub(crate) generated_token_fingerprint: u64,
}

/// Applies the non-MLX release gate after all five isolated measurement cells finish.
///
/// Returning a stable reason instead of panicking here lets the hermetic suite prove both accepted
/// and rejected evidence. The ignored named-model test converts any rejection into its assertion.
pub(crate) fn validate_mtp_depth_release_gate(
    target_only_before: &MtpDepthMeasurement,
    target_only_after: &MtpDepthMeasurement,
    depth_one: &MtpDepthMeasurement,
    depth_two: &MtpDepthMeasurement,
    depth_three: &MtpDepthMeasurement,
    expected_output_token_count: usize,
) -> Result<(), &'static str> {
    if target_only_before.generated_token_fingerprint
        != target_only_after.generated_token_fingerprint
    {
        return Err("target-only controls produced different token sequences");
    }

    let all_measurements = [
        target_only_before,
        target_only_after,
        depth_one,
        depth_two,
        depth_three,
    ];
    for measurement in all_measurements {
        if measurement.output_token_count != expected_output_token_count {
            return Err("a acceptance cell did not generate the required token count");
        }
        if measurement.maximum_active_mlx_memory_bytes > measurement.mlx_memory_ceiling_bytes {
            return Err("a acceptance cell exceeded its active MLX memory ceiling");
        }
        let allowed_peak_bytes = measurement
            .mlx_memory_ceiling_bytes
            .saturating_add(measurement.mlx_memory_ceiling_bytes / 100);
        if measurement.maximum_peak_mlx_memory_bytes > allowed_peak_bytes {
            return Err("a acceptance cell exceeded its approved MLX peak allowance");
        }
        if measurement.operational_fallback_count != 0 {
            return Err("a acceptance cell used operational target-only fallback");
        }
    }

    let target_only_total_request_seconds = target_only_before
        .total_request_elapsed_seconds
        .min(target_only_after.total_request_elapsed_seconds);
    for (expected_depth, measurement) in [(1_u8, depth_one), (2, depth_two), (3, depth_three)] {
        if measurement.draft_depth != Some(expected_depth) {
            return Err("an MTP acceptance cell reported the wrong fixed depth");
        }
        if measurement.generated_token_fingerprint != target_only_before.generated_token_fingerprint
        {
            return Err("an MTP depth changed the target-authoritative token sequence");
        }
        if measurement.proposed_draft_count == 0
            || measurement.effective_depth_total < u64::from(expected_depth)
        {
            return Err("an MTP acceptance cell did not exercise its requested depth");
        }
        if measurement.total_request_elapsed_seconds >= target_only_total_request_seconds {
            return Err("an MTP depth did not improve end-to-end request latency");
        }
    }
    if depth_two.total_request_elapsed_seconds >= depth_one.total_request_elapsed_seconds {
        return Err("MTP depth two did not improve on depth one");
    }
    if depth_three.total_request_elapsed_seconds >= depth_one.total_request_elapsed_seconds
        || depth_three.total_request_elapsed_seconds >= depth_two.total_request_elapsed_seconds
    {
        return Err("MTP depth three did not improve on every shallower depth");
    }
    Ok(())
}
