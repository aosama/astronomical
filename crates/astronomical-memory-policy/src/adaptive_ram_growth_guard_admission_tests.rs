//! Verbalized-sampling portfolio for adaptive RAM growth:
//! ordinary fitting growth (0.55), rising transient pressure (0.20), exact-limit
//! boundary (0.10), arithmetic overflow (0.08), lower later spike (0.05), and
//! active-memory release between observations (0.02). Probabilities are test-design
//! estimates rather than measured production frequencies.

use crate::budget::adaptive_growth::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    AdaptiveRamGrowthGuardError,
};
use crate::budget::adaptive_growth_projection::TransientWorkspaceReserveSource;
use crate::phase::MemoryPhase;

const DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident);
#[test]
fn should_reject_a_zero_active_memory_limit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let guard_creation_error = AdaptiveRamGrowthGuard::new(0)
        .expect_err("a zero-byte active-memory limit cannot admit useful growth");

    assert_eq!(
        guard_creation_error,
        AdaptiveRamGrowthGuardError::InvalidActiveMemoryCeiling
    );
}

#[test]
fn should_allow_unobserved_growth_when_exact_persistent_bytes_fit_the_limit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 700, 300, 0, 0, 0)
        .expect("exact persistent growth ending at the limit should not overflow");

    assert!(projection.fits_stable_and_peak_limits());
}

#[test]
fn should_apply_transient_learning_across_unseen_contexts_for_admission() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let observed_prefill_context =
        AdaptiveRamGrowthContext::prefill(128, 0, false, AdaptiveRamGrowthExecutionProfile::Paged);
    let different_prefill_context =
        AdaptiveRamGrowthContext::prefill(256, 0, false, AdaptiveRamGrowthExecutionProfile::Paged);

    adaptive_ram_growth_guard.record_completed_growth_for_context(
        observed_prefill_context,
        true,
        400,
        500,
        700,
        0,
    );

    let observed_context_projection = adaptive_ram_growth_guard
        .project_growth_for_context(observed_prefill_context, 700, 100, 0, 0, 0)
        .expect("the observed context should project without overflow");
    let different_context_projection = adaptive_ram_growth_guard
        .project_growth_for_context(different_prefill_context, 700, 100, 0, 0, 0)
        .expect("an unseen context should project without overflow");

    assert_eq!(
        observed_context_projection.observed_transient_high_water_bytes(),
        200
    );
    // An unseen LARGER shape keeps the smaller shape's window unscaled as a
    // lower bound (the analytic floor is zero here). Doubling it to 400 bytes
    // by the token ratio multiplied fixed per-forward overhead and once
    // projected a 23 GB phantom transient (measured 2026-10-24).
    assert_eq!(
        different_context_projection.observed_transient_high_water_bytes(),
        200
    );
    assert_eq!(
        different_context_projection.transient_reserve_source(),
        TransientWorkspaceReserveSource::BeyondObservedTokenSpan,
    );
    assert_eq!(observed_context_projection.stable_projected_bytes(), 800);
    assert_eq!(observed_context_projection.peak_projected_bytes(), 1_000);
    assert_eq!(
        observed_context_projection.allowed_active_memory_bytes(),
        1_010
    );
}

#[test]
fn should_keep_exact_context_evidence_separate_from_phase_admission_maximum() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let smaller_context =
        AdaptiveRamGrowthContext::prefill(128, 0, false, AdaptiveRamGrowthExecutionProfile::Paged);
    let larger_context =
        AdaptiveRamGrowthContext::prefill(256, 0, false, AdaptiveRamGrowthExecutionProfile::Paged);
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        smaller_context,
        true,
        100,
        100,
        150,
        0,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        larger_context,
        true,
        100,
        100,
        300,
        0,
    );

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes_for_context(smaller_context),
        50
    );
    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        200
    );
}

#[test]
fn should_accept_stable_memory_at_c_and_reject_one_byte_above_c() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let adaptive_ram_growth_context =
        AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident);

    let fitting_projection = adaptive_ram_growth_guard
        .project_growth_for_context(adaptive_ram_growth_context, 700, 300, 0, 0, 0)
        .expect("stable growth ending exactly at C should project");
    let exceeding_projection = adaptive_ram_growth_guard
        .project_growth_for_context(adaptive_ram_growth_context, 700, 301, 0, 0, 0)
        .expect("stable growth one byte above C should project");

    assert_eq!(fitting_projection.stable_projected_bytes(), 1_000);
    assert!(fitting_projection.fits_stable_and_peak_limits());
    assert_eq!(exceeding_projection.stable_projected_bytes(), 1_001);
    assert_eq!(
        exceeding_projection.operation_reclamation_required_bytes(),
        1
    );
    assert!(!exceeding_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_accept_peak_memory_at_p_and_reject_one_byte_above_p() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let fitting_peak_context =
        AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident);
    let exceeding_peak_context =
        AdaptiveRamGrowthContext::decode(2, AdaptiveRamGrowthExecutionProfile::Resident);
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        fitting_peak_context,
        true,
        0,
        0,
        10,
        0,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        exceeding_peak_context,
        true,
        0,
        0,
        11,
        0,
    );

    let fitting_projection = adaptive_ram_growth_guard
        .project_growth_for_context(fitting_peak_context, 900, 100, 0, 0, 0)
        .expect("peak ending exactly at P should project");
    let exceeding_projection = adaptive_ram_growth_guard
        .project_growth_for_context(exceeding_peak_context, 900, 101, 0, 0, 0)
        .expect("peak one byte above P should project");

    // Issue #623: each context reserves its own exact evidence (10 and 11),
    // not the global maximum. The fitting context therefore peaks exactly at
    // P = C + 1 percent and is admitted; one byte more requires reclamation.
    assert_eq!(fitting_projection.peak_projected_bytes(), 1_010);
    assert!(fitting_projection.fits_stable_and_peak_limits());
    assert_eq!(exceeding_projection.peak_projected_bytes(), 1_012);
    assert_eq!(
        exceeding_projection.operation_reclamation_required_bytes(),
        2
    );
    assert!(!exceeding_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_not_retain_a_final_partial_prefill_tail_as_reusable_evidence() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let partial_tail_context =
        AdaptiveRamGrowthContext::prefill(37, 0, false, AdaptiveRamGrowthExecutionProfile::Paged);
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        partial_tail_context,
        false,
        0,
        0,
        500,
        0,
    );

    assert_eq!(
        adaptive_ram_growth_guard
            .project_growth_for_context(partial_tail_context, 400, 100, 0, 0, 0)
            .expect("an unretained tail should project")
            .observed_transient_high_water_bytes(),
        0
    );
}

#[test]
fn should_reject_growth_when_the_peak_projection_exceeds_the_limit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        400,
        500,
        650,
        0,
    );

    // peak: 800 + 100 + 150 = 1_050 > P=1_010
    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 800, 100, 0, 0, 0)
        .expect("the peak projection should not overflow");

    assert_eq!(projection.peak_projected_bytes(), 1_050);
    assert_eq!(projection.operation_reclamation_required_bytes(), 40);
    assert!(!projection.fits_stable_and_peak_limits());
}

#[test]
fn should_reject_an_overflowing_memory_projection() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(usize::MAX)
        .expect("the platform maximum should be a valid positive limit");

    let growth_rejection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, usize::MAX, 1, 0, 0, 0)
        .expect_err("an overflowing byte projection must not be admitted");

    assert_eq!(
        growth_rejection,
        AdaptiveRamGrowthGuardError::MemoryProjectionOverflow
    );
}

#[test]
fn should_preserve_the_highest_transient_observation_after_a_lower_spike() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        400,
        500,
        700,
        0,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        500,
        550,
        600,
        0,
    );

    // peak: 701 + 100 + 200 = 1_001 <= P=1_010
    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 701, 100, 0, 0, 0)
        .expect("a high-water projection within the platform range should not overflow");

    assert_eq!(projection.observed_transient_high_water_bytes(), 200);
    assert_eq!(projection.peak_projected_bytes(), 1_001);
    assert_eq!(projection.operation_reclamation_required_bytes(), 0);
    assert!(projection.fits_stable_and_peak_limits());
}

#[test]
fn should_not_underflow_when_active_memory_falls_without_a_new_allocator_peak() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        700,
        500,
        600,
        0,
    );

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 800, 200, 0, 0, 0)
        .expect("falling active memory should not overflow the projection");

    assert_eq!(projection.observed_transient_high_water_bytes(), 0);
    assert!(projection.fits_stable_and_peak_limits());
}
