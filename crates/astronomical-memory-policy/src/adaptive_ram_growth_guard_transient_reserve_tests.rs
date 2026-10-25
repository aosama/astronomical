//! Transient-reserve resolution for forwards larger than any observed forward.
//!
//! Replays the 2026-10-24 incident: a fully resident 4-bit model under a
//! 27 GB ceiling observed a 38-token and a 1-token prefill chunk, then admitted
//! a 2,048-token chunk. Scaling the 1-token window up by the token ratio
//! projected a 23 GB phantom transient, the resident engine requested the
//! streaming engine, and the model served from SSD with over 5 GB unused.

use crate::budget::adaptive_growth::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};
use crate::budget::adaptive_growth_projection::TransientWorkspaceReserveSource;
use crate::budget::transient_workspace_reserve::scale_transient_to_token_count;

const TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES: usize = 28_991_029_248;
const THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES: usize = 28_113_460;
const ONE_TOKEN_PREFILL_TRANSIENT_BYTES: usize = 11_255_538;
const THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES: usize = 2_566_914_048;
const RESIDENT_ACTIVE_MEMORY_BEFORE_LONG_CHUNK_BYTES: usize = 22_164_699_918;
const LONG_CHUNK_PERSISTENT_GROWTH_BYTES: usize = 106_332_160;
const LONG_PROMPT_CHUNK_TOKEN_COUNT: usize = 2_048;

fn resident_prefill_context(forward_token_count: usize) -> AdaptiveRamGrowthContext {
    AdaptiveRamGrowthContext::prefill(
        forward_token_count,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Resident,
    )
}

fn record_transient_window(
    adaptive_ram_growth_guard: &mut AdaptiveRamGrowthGuard,
    adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    transient_window_bytes: usize,
) {
    // Zero before/after samples make the peak sample the stored residual.
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        adaptive_ram_growth_context,
        true,
        0,
        0,
        transient_window_bytes,
        0,
    );
}

fn guard_after_the_short_prompt() -> AdaptiveRamGrowthGuard {
    let mut adaptive_ram_growth_guard =
        AdaptiveRamGrowthGuard::new(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the reproduced 27 GB ceiling should create a guard");
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        resident_prefill_context(38),
        THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES,
    );
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        resident_prefill_context(1),
        ONE_TOKEN_PREFILL_TRANSIENT_BYTES,
    );
    adaptive_ram_growth_guard
}

#[test]
fn should_keep_a_resident_long_prompt_chunk_inside_the_ceiling_after_a_one_token_prefill_observation()
 {
    let adaptive_ram_growth_guard = guard_after_the_short_prompt();

    let long_chunk_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
            RESIDENT_ACTIVE_MEMORY_BEFORE_LONG_CHUNK_BYTES,
            LONG_CHUNK_PERSISTENT_GROWTH_BYTES,
            0,
            0,
            THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
        )
        .expect("the reproduced long-chunk projection should fit the platform range");

    // Before the fix: 11,255,538 × 2,048 = 23,051,341,824 reserved bytes and a
    // 45,322,373,902-byte peak against 29,280,939,540 allowed bytes.
    assert_eq!(
        long_chunk_projection.observed_transient_high_water_bytes(),
        THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES
    );
    assert_eq!(
        long_chunk_projection.transient_reserve_source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan,
    );
    assert_eq!(long_chunk_projection.peak_projected_bytes(), 24_837_946_126);
    assert_eq!(
        long_chunk_projection.allowed_active_memory_bytes(),
        29_280_939_540
    );
    assert!(long_chunk_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_never_scale_a_smaller_observation_up_by_the_token_ratio() {
    let adaptive_ram_growth_guard = guard_after_the_short_prompt();

    let long_chunk_reserve = adaptive_ram_growth_guard.admission_transient_reserve_for_context(
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        0,
    );

    assert_eq!(
        long_chunk_reserve.reserve_bytes(),
        THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES
    );
    assert_eq!(
        long_chunk_reserve.source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan,
    );
}

#[test]
fn should_prefer_an_unscaled_smaller_observation_that_exceeds_the_analytic_floor() {
    let mut adaptive_ram_growth_guard =
        AdaptiveRamGrowthGuard::new(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the reproduced 27 GB ceiling should create a guard");
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        resident_prefill_context(1_024),
        3_000_000_000,
    );

    let long_chunk_reserve = adaptive_ram_growth_guard.admission_transient_reserve_for_context(
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
    );

    assert_eq!(long_chunk_reserve.reserve_bytes(), 3_000_000_000);
    assert_eq!(
        long_chunk_reserve.source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan,
    );
}

#[test]
fn should_trust_covering_evidence_over_the_analytic_floor() {
    let mut adaptive_ram_growth_guard =
        AdaptiveRamGrowthGuard::new(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the reproduced 27 GB ceiling should create a guard");
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        AdaptiveRamGrowthContext::prefill(
            LONG_PROMPT_CHUNK_TOKEN_COUNT,
            3,
            false,
            AdaptiveRamGrowthExecutionProfile::Resident,
        ),
        1_500_000_000,
    );

    // Same token count in another position bucket is covering evidence at a
    // one-to-one ratio; measured evidence outranks the analytic estimate.
    let long_chunk_reserve = adaptive_ram_growth_guard.admission_transient_reserve_for_context(
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
    );

    assert_eq!(long_chunk_reserve.reserve_bytes(), 1_500_000_000);
    assert_eq!(
        long_chunk_reserve.source(),
        TransientWorkspaceReserveSource::PhaseScaled,
    );
}

#[test]
fn should_reserve_the_analytic_floor_when_no_forward_has_completed() {
    let adaptive_ram_growth_guard =
        AdaptiveRamGrowthGuard::new(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the reproduced 27 GB ceiling should create a guard");

    let first_chunk_reserve = adaptive_ram_growth_guard.admission_transient_reserve_for_context(
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
    );

    assert_eq!(
        first_chunk_reserve.reserve_bytes(),
        THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES
    );
    assert_eq!(
        first_chunk_reserve.source(),
        TransientWorkspaceReserveSource::GlobalMaximum,
    );
}

#[test]
fn should_use_the_global_maximum_for_a_zero_token_forward() {
    let adaptive_ram_growth_guard = guard_after_the_short_prompt();

    let zero_token_reserve = adaptive_ram_growth_guard
        .admission_transient_reserve_for_context(resident_prefill_context(0), 0);

    assert_eq!(
        zero_token_reserve.reserve_bytes(),
        THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES
    );
    assert_eq!(
        zero_token_reserve.source(),
        TransientWorkspaceReserveSource::GlobalMaximum,
    );
}

#[test]
fn should_count_a_zero_token_observation_only_toward_the_global_maximum() {
    let mut adaptive_ram_growth_guard =
        AdaptiveRamGrowthGuard::new(TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES)
            .expect("the reproduced 27 GB ceiling should create a guard");
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        resident_prefill_context(0),
        50_000,
    );

    let long_chunk_reserve = adaptive_ram_growth_guard.admission_transient_reserve_for_context(
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        0,
    );

    assert_eq!(long_chunk_reserve.reserve_bytes(), 50_000);
    assert_eq!(
        long_chunk_reserve.source(),
        TransientWorkspaceReserveSource::GlobalMaximum,
    );
}

#[test]
fn should_still_reject_a_forward_whose_own_measured_window_does_not_fit() {
    let mut adaptive_ram_growth_guard = guard_after_the_short_prompt();
    record_transient_window(
        &mut adaptive_ram_growth_guard,
        resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
        8_000_000_000,
    );

    let long_chunk_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
            RESIDENT_ACTIVE_MEMORY_BEFORE_LONG_CHUNK_BYTES,
            LONG_CHUNK_PERSISTENT_GROWTH_BYTES,
            0,
            0,
            THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
        )
        .expect("the measured long-chunk projection should fit the platform range");

    assert_eq!(
        long_chunk_projection.transient_reserve_source(),
        TransientWorkspaceReserveSource::ExactContext,
    );
    assert!(!long_chunk_projection.fits_stable_and_peak_limits());
    // Only the transient peak is short, so a smaller chunk can still run
    // resident; stable ownership alone fits the ceiling.
    assert!(long_chunk_projection.fits_stable_limit());
}

#[test]
fn should_report_a_stable_shortfall_that_no_smaller_forward_can_fix() {
    let adaptive_ram_growth_guard = guard_after_the_short_prompt();

    let overfull_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
            TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES,
            LONG_CHUNK_PERSISTENT_GROWTH_BYTES,
            0,
            0,
            THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
        )
        .expect("the overfull projection should fit the platform range");

    assert!(!overfull_projection.fits_stable_limit());
    assert!(!overfull_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_fit_the_stable_limit_at_exactly_the_ceiling() {
    let adaptive_ram_growth_guard = guard_after_the_short_prompt();

    let boundary_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            resident_prefill_context(LONG_PROMPT_CHUNK_TOKEN_COUNT),
            TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES - LONG_CHUNK_PERSISTENT_GROWTH_BYTES,
            LONG_CHUNK_PERSISTENT_GROWTH_BYTES,
            0,
            0,
            THREE_COMPLETE_EXPERT_LAYER_PREFILL_FLOOR_BYTES,
        )
        .expect("the boundary projection should fit the platform range");

    assert_eq!(
        boundary_projection.stable_projected_bytes(),
        TWENTY_SEVEN_GIGABYTE_ACTIVE_MEMORY_CEILING_BYTES
    );
    assert!(boundary_projection.fits_stable_limit());
    assert!(!boundary_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_refuse_to_scale_an_observation_up_to_a_larger_forward() {
    assert_eq!(
        scale_transient_to_token_count(
            ONE_TOKEN_PREFILL_TRANSIENT_BYTES,
            1,
            LONG_PROMPT_CHUNK_TOKEN_COUNT,
        ),
        None
    );
}

#[test]
fn should_scale_a_larger_observation_down_with_ceiling_division() {
    assert_eq!(scale_transient_to_token_count(10, 3, 1), Some(4));
    assert_eq!(
        scale_transient_to_token_count(THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES, 38, 38),
        Some(THIRTY_EIGHT_TOKEN_PREFILL_TRANSIENT_BYTES)
    );
}

#[test]
fn should_refuse_to_scale_with_a_zero_token_count() {
    assert_eq!(scale_transient_to_token_count(10, 0, 1), None);
    assert_eq!(scale_transient_to_token_count(10, 3, 0), None);
}
