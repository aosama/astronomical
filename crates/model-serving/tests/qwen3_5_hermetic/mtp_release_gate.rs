use crate::common::mtp_depth_release_gate::{MtpDepthMeasurement, validate_mtp_depth_release_gate};

const EXPECTED_OUTPUT_TOKEN_COUNT: usize = 1_024;
const TARGET_TOKEN_FINGERPRINT: u64 = 0xA57A_0A1C;

fn measurement(
    cell_name: &str,
    draft_depth: Option<u8>,
    total_request_elapsed_seconds: f64,
) -> MtpDepthMeasurement {
    let exercised_depth = draft_depth.unwrap_or(0);
    MtpDepthMeasurement {
        cell_name: cell_name.to_owned(),
        draft_depth,
        output_token_count: EXPECTED_OUTPUT_TOKEN_COUNT,
        generation_elapsed_seconds: total_request_elapsed_seconds - 1.0,
        total_request_elapsed_seconds,
        tokens_per_second: 100.0,
        maximum_active_mlx_memory_bytes: 900,
        maximum_peak_mlx_memory_bytes: 1_000,
        mlx_memory_ceiling_bytes: 1_000,
        operational_fallback_count: 0,
        proposed_draft_count: u64::from(exercised_depth) * 100,
        effective_depth_total: u64::from(exercised_depth) * 100,
        generated_token_fingerprint: TARGET_TOKEN_FINGERPRINT,
    }
}

fn passing_measurements() -> [MtpDepthMeasurement; 5] {
    // The synthetic controls bracket drift while each deeper depth improves end-to-end latency.
    [
        measurement("target_only_before", None, 10.0),
        measurement("target_only_after", None, 10.2),
        measurement("depth_one", Some(1), 9.0),
        measurement("depth_two", Some(2), 8.0),
        measurement("depth_three", Some(3), 7.0),
    ]
}

#[test]
fn should_accept_target_authoritative_memory_safe_end_to_end_depth_evidence() {
    let [
        target_before,
        target_after,
        depth_one,
        depth_two,
        depth_three,
    ] = passing_measurements();

    validate_mtp_depth_release_gate(
        &target_before,
        &target_after,
        &depth_one,
        &depth_two,
        &depth_three,
        EXPECTED_OUTPUT_TOKEN_COUNT,
    )
    .expect("complete target-authoritative depth evidence should pass the release gate");
}

#[test]
fn should_reject_a_depth_that_changes_target_authoritative_output() {
    let [
        target_before,
        target_after,
        depth_one,
        depth_two,
        mut depth_three,
    ] = passing_measurements();
    depth_three.generated_token_fingerprint = TARGET_TOKEN_FINGERPRINT.wrapping_add(1);

    let rejection = validate_mtp_depth_release_gate(
        &target_before,
        &target_after,
        &depth_one,
        &depth_two,
        &depth_three,
        EXPECTED_OUTPUT_TOKEN_COUNT,
    )
    .expect_err("changed model output must block production qualification");

    assert_eq!(
        rejection,
        "an MTP depth changed the target-authoritative token sequence"
    );
}

#[test]
fn should_reject_depth_three_when_it_does_not_beat_every_shallower_control() {
    let [
        target_before,
        target_after,
        depth_one,
        depth_two,
        mut depth_three,
    ] = passing_measurements();
    depth_three.total_request_elapsed_seconds = 8.5;

    let rejection = validate_mtp_depth_release_gate(
        &target_before,
        &target_after,
        &depth_one,
        &depth_two,
        &depth_three,
        EXPECTED_OUTPUT_TOKEN_COUNT,
    )
    .expect_err("a slower deeper route must remain unqualified");

    assert_eq!(
        rejection,
        "MTP depth three did not improve on every shallower depth"
    );
}
