//! Verbalized-sampling portfolio for adaptive RAM growth:
//! ordinary fitting growth (0.55), rising transient pressure (0.20), exact-limit
//! boundary (0.10), arithmetic overflow (0.08), lower later spike (0.05), and
//! active-memory release between observations (0.02). Probabilities are test-design
//! estimates rather than measured production frequencies.

use crate::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    AdaptiveRamGrowthGuardError, AdaptiveRamGrowthTransientReserveSource,
};

const DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Resident);
#[test]
fn should_admit_growth_without_expert_reclamation_when_only_the_recovery_reserve_is_short() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    // Learn a 150-byte transient window.
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        400,
        500,
        650,
        0,
    );

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 600, 150, 0, 0)
        .expect("a valid projection should not overflow");

    assert_eq!(projection.current_active_memory_bytes(), 600);
    assert_eq!(projection.exact_persistent_growth_bytes(), 150);
    assert_eq!(projection.observed_transient_high_water_bytes(), 150);
    assert_eq!(projection.peak_projected_bytes(), 900);
    assert_eq!(projection.recovery_projected_bytes(), 1_050);
    assert_eq!(projection.active_memory_ceiling_bytes(), 1_000);
    assert_eq!(projection.operation_reclamation_required_bytes(), 0);
    assert_eq!(projection.recovery_reserve_shortfall_bytes(), 40);
    assert_eq!(
        projection
            .expert_retention_reclamation_plan(1_000)
            .reclamation_target_bytes(),
        0,
        "recovery-only headroom must not force expert reclamation before a typed allocation failure"
    );
    assert!(projection.fits_stable_and_peak_limits());
    assert!(!projection.has_full_recovery_reserve());
}

#[test]
fn should_report_only_the_measured_peak_shortfall_as_required_reclamation() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    // Learn a 200-byte transient window.
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        400,
        500,
        700,
        0,
    );

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 700, 150, 0, 0)
        .expect("a valid projection should not overflow");

    // peak: 700 + 150 + 200 = 1_050, deficit against P=1,010 is 40.
    // recovery: 1_050 + 200 = 1_250, shortfall against P=1,010 is 240.
    assert_eq!(projection.operation_reclamation_required_bytes(), 40);
    assert_eq!(projection.recovery_reserve_shortfall_bytes(), 240);
    assert_eq!(
        projection
            .expert_retention_reclamation_plan(1_000)
            .reclamation_target_bytes(),
        40,
        "preemptive reclamation should cover the measured peak shortfall, not the larger diagnostic recovery shortfall"
    );
    assert!(!projection.fits_stable_and_peak_limits());
    assert!(!projection.has_full_recovery_reserve());
}

#[test]
fn should_reject_a_recovery_projection_overflow() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(usize::MAX)
        .expect("the platform maximum should be a valid positive limit");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        0,
        0,
        usize::MAX,
        0,
    );

    let projection_error = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 0, 0, 0, 0)
        .expect_err("an overflowing recovery projection must fail closed");

    assert_eq!(
        projection_error,
        AdaptiveRamGrowthGuardError::MemoryProjectionOverflow
    );
}

// Issue #623: one large shape's transient high-water was charged unchanged to
// every later forward of every other shape, demoting a fully resident expert
// owner whose own chunk needed a fraction of the reserved bytes. The admission
// reserve must therefore be shaped by the forward being admitted, with the
// global maximum kept only for a phase that has no evidence at all.
#[test]
fn should_scale_an_unobserved_shape_reserve_by_its_own_token_count() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000_000)
        .expect("a positive active-memory limit should create a guard");
    let large_paged_prefill_context = AdaptiveRamGrowthContext::prefill(
        8_192,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Paged,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        large_paged_prefill_context,
        true,
        0,
        0,
        8_000,
        0,
    );

    let small_resident_prefill_context = AdaptiveRamGrowthContext::prefill(
        2_048,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Resident,
    );
    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(small_resident_prefill_context, 900_000, 10_000, 0, 0)
        .expect("the smaller shape should project without overflow");

    // 8,000 bytes observed at 8,192 tokens scale to 2,000 bytes at 2,048 tokens
    // (ceiling division), not the borrowed 8,000-byte absolute window.
    assert_eq!(projection.observed_transient_high_water_bytes(), 2_000);
    assert_eq!(
        projection.transient_reserve_source(),
        AdaptiveRamGrowthTransientReserveSource::PhaseScaled,
    );
}

#[test]
fn should_prefer_exact_context_evidence_over_the_scaled_phase_estimate() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000_000)
        .expect("a positive active-memory limit should create a guard");
    let large_paged_prefill_context = AdaptiveRamGrowthContext::prefill(
        8_192,
        0,
        false,
        AdaptiveRamGrowthExecutionProfile::Paged,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        large_paged_prefill_context,
        true,
        0,
        0,
        8_000,
        0,
    );
    let exact_resident_prefill_context = AdaptiveRamGrowthContext::prefill(
        2_048,
        3,
        false,
        AdaptiveRamGrowthExecutionProfile::Resident,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        exact_resident_prefill_context,
        true,
        0,
        0,
        1_500,
        0,
    );

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(exact_resident_prefill_context, 900_000, 10_000, 0, 0)
        .expect("the exact context should project without overflow");

    assert_eq!(projection.observed_transient_high_water_bytes(), 1_500);
    assert_eq!(
        projection.transient_reserve_source(),
        AdaptiveRamGrowthTransientReserveSource::ExactContext,
    );
}

#[test]
fn should_use_the_largest_scaled_phase_observation_for_an_unobserved_shape() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        AdaptiveRamGrowthContext::prefill(
            8_192,
            0,
            false,
            AdaptiveRamGrowthExecutionProfile::Paged,
        ),
        true,
        0,
        0,
        8_000,
        0,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        AdaptiveRamGrowthContext::prefill(
            1_024,
            9,
            false,
            AdaptiveRamGrowthExecutionProfile::Resident,
        ),
        true,
        0,
        0,
        3_000,
        0,
    );

    // 8,000 × 2,048/8,192 = 2,000; 3,000 × 2,048/1,024 = 6,000. The largest
    // scaled observation is the conservative proportional estimate.
    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::prefill(
                2_048,
                0,
                false,
                AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            900_000,
            10_000,
            0,
            0,
        )
        .expect("the unobserved shape should project without overflow");

    assert_eq!(projection.observed_transient_high_water_bytes(), 6_000);
    assert_eq!(
        projection.transient_reserve_source(),
        AdaptiveRamGrowthTransientReserveSource::PhaseScaled,
    );
}

#[test]
fn should_keep_the_global_maximum_for_a_phase_without_any_evidence() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        0,
        0,
        500,
        0,
    );

    // Decode evidence exists but prefill has none. The first prefill reserves
    // the global maximum; its completion records prefill evidence and every
    // later prefill reserves proportionally.
    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::prefill(
                2_048,
                0,
                false,
                AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            900_000,
            10_000,
            0,
            0,
        )
        .expect("the first prefill should project without overflow");

    assert_eq!(projection.observed_transient_high_water_bytes(), 500);
    assert_eq!(
        projection.transient_reserve_source(),
        AdaptiveRamGrowthTransientReserveSource::GlobalMaximum,
    );
}
