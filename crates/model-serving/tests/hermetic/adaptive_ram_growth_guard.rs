//! Verbalized-sampling portfolio for adaptive RAM growth:
//! ordinary fitting growth (0.55), rising transient pressure (0.20), exact-limit
//! boundary (0.10), arithmetic overflow (0.08), lower later spike (0.05), and
//! active-memory release between observations (0.02). Probabilities are test-design
//! estimates rather than measured production frequencies.

use astronomical_model_serving::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthPhase,
};

const DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::decode(1, false, false);
const DEFAULT_PREFILL_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::prefill(128, 0, false, false, false);

#[test]
fn should_reject_a_zero_active_memory_limit() {
    let guard_creation_error = AdaptiveRamGrowthGuard::new(0)
        .expect_err("a zero-byte active-memory limit cannot admit useful growth");

    assert_eq!(
        guard_creation_error,
        AdaptiveRamGrowthGuardError::InvalidActiveMemoryLimit
    );
}

#[test]
fn should_allow_unobserved_growth_when_exact_persistent_bytes_fit_the_limit() {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 700, 300, 0, 0)
        .expect("exact persistent growth ending at the limit should not overflow");

    assert!(projection.fits_stable_and_peak_limits());
}

#[test]
fn should_apply_transient_learning_only_to_the_exact_execution_context() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let observed_prefill_context = AdaptiveRamGrowthContext::prefill(128, 0, false, false, true);
    let different_prefill_context = AdaptiveRamGrowthContext::prefill(256, 0, false, false, true);

    adaptive_ram_growth_guard.record_completed_growth_for_context(
        observed_prefill_context,
        true,
        400,
        500,
        700,
        0,
    );

    let observed_context_projection = adaptive_ram_growth_guard
        .project_growth_for_context(observed_prefill_context, 700, 100, 0, 0)
        .expect("the observed context should project without overflow");
    let different_context_projection = adaptive_ram_growth_guard
        .project_growth_for_context(different_prefill_context, 700, 100, 0, 0)
        .expect("an unseen context should project without overflow");

    assert_eq!(
        observed_context_projection.observed_transient_high_water_bytes(),
        200
    );
    assert_eq!(
        different_context_projection.observed_transient_high_water_bytes(),
        0
    );
    assert_eq!(observed_context_projection.stable_projected_bytes(), 800);
    assert_eq!(observed_context_projection.peak_projected_bytes(), 1_000);
    assert_eq!(
        observed_context_projection.allowed_active_memory_bytes(),
        1_010
    );
}

#[test]
fn should_accept_stable_memory_at_c_and_reject_one_byte_above_c() {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let adaptive_ram_growth_context = AdaptiveRamGrowthContext::decode(1, false, false);

    let fitting_projection = adaptive_ram_growth_guard
        .project_growth_for_context(adaptive_ram_growth_context, 700, 300, 0, 0)
        .expect("stable growth ending exactly at C should project");
    let exceeding_projection = adaptive_ram_growth_guard
        .project_growth_for_context(adaptive_ram_growth_context, 700, 301, 0, 0)
        .expect("stable growth one byte above C should project");

    assert_eq!(fitting_projection.stable_projected_bytes(), 1_000);
    assert!(fitting_projection.fits_stable_and_peak_limits());
    assert_eq!(exceeding_projection.stable_projected_bytes(), 1_001);
    assert_eq!(exceeding_projection.required_reclamation_bytes(), 1);
    assert!(!exceeding_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_accept_peak_memory_at_p_and_reject_one_byte_above_p() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let fitting_peak_context = AdaptiveRamGrowthContext::decode(1, false, false);
    let exceeding_peak_context = AdaptiveRamGrowthContext::decode(2, false, false);
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
        .project_growth_for_context(fitting_peak_context, 900, 100, 0, 0)
        .expect("peak ending exactly at P should project");
    let exceeding_projection = adaptive_ram_growth_guard
        .project_growth_for_context(exceeding_peak_context, 900, 100, 0, 0)
        .expect("peak one byte above P should project");

    assert_eq!(fitting_projection.peak_projected_bytes(), 1_010);
    assert!(fitting_projection.fits_stable_and_peak_limits());
    assert_eq!(exceeding_projection.peak_projected_bytes(), 1_011);
    assert_eq!(exceeding_projection.required_reclamation_bytes(), 1);
    assert!(!exceeding_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_keep_prompt_position_visual_mtp_and_expert_mode_contexts_independent() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let observed_prefill_context = AdaptiveRamGrowthContext::prefill(128, 7, true, true, true);
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        observed_prefill_context,
        true,
        0,
        0,
        200,
        0,
    );

    for independent_prefill_context in [
        AdaptiveRamGrowthContext::prefill(128, 8, true, true, true),
        AdaptiveRamGrowthContext::prefill(128, 7, false, true, true),
        AdaptiveRamGrowthContext::prefill(128, 7, true, false, true),
        AdaptiveRamGrowthContext::prefill(128, 7, true, true, false),
    ] {
        assert_eq!(
            adaptive_ram_growth_guard
                .project_growth_for_context(independent_prefill_context, 500, 100, 0, 0)
                .expect("an independent context should project")
                .observed_transient_high_water_bytes(),
            0
        );
    }
}

#[test]
fn should_not_retain_a_final_partial_prefill_tail_as_reusable_evidence() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let partial_tail_context = AdaptiveRamGrowthContext::prefill(37, 0, false, false, true);
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
            .project_growth_for_context(partial_tail_context, 400, 100, 0, 0)
            .expect("an unretained tail should project")
            .observed_transient_high_water_bytes(),
        0
    );
}

#[test]
fn should_reject_growth_when_the_peak_projection_exceeds_the_limit() {
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
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 800, 100, 0, 0)
        .expect("the peak projection should not overflow");

    assert_eq!(projection.peak_projected_bytes(), 1_050);
    assert_eq!(projection.required_reclamation_bytes(), 40);
    assert!(!projection.fits_stable_and_peak_limits());
}

#[test]
fn should_reject_an_overflowing_memory_projection() {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(usize::MAX)
        .expect("the platform maximum should be a valid positive limit");

    let growth_rejection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, usize::MAX, 1, 0, 0)
        .expect_err("an overflowing byte projection must not be admitted");

    assert_eq!(
        growth_rejection,
        AdaptiveRamGrowthGuardError::MemoryProjectionOverflow
    );
}

#[test]
fn should_preserve_the_highest_transient_observation_after_a_lower_spike() {
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
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 701, 100, 0, 0)
        .expect("a high-water projection within the platform range should not overflow");

    assert_eq!(projection.observed_transient_high_water_bytes(), 200);
    assert_eq!(projection.peak_projected_bytes(), 1_001);
    assert_eq!(projection.required_reclamation_bytes(), 0);
    assert!(projection.fits_stable_and_peak_limits());
}

#[test]
fn should_not_underflow_when_active_memory_falls_without_a_new_allocator_peak() {
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
        .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 800, 200, 0, 0)
        .expect("falling active memory should not overflow the projection");

    assert_eq!(projection.observed_transient_high_water_bytes(), 0);
    assert!(projection.fits_stable_and_peak_limits());
}

#[test]
fn should_allow_growth_when_the_measured_peak_fits_but_the_soft_recovery_reserve_does_not() {
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
    assert_eq!(projection.soft_recovery_projected_bytes(), 1_050);
    assert_eq!(projection.active_memory_limit_bytes(), 1_000);
    assert_eq!(projection.required_reclamation_bytes(), 0);
    assert_eq!(projection.soft_reserve_shortfall_bytes(), 40);
    assert!(projection.fits_stable_and_peak_limits());
    assert!(!projection.has_full_recovery_reserve());
}

#[test]
fn should_report_only_the_measured_peak_shortfall_as_required_reclamation() {
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
    // soft: 1_050 + 200 = 1_250, shortfall against P=1,010 is 240.
    assert_eq!(projection.required_reclamation_bytes(), 40);
    assert_eq!(projection.soft_reserve_shortfall_bytes(), 240);
    assert!(!projection.fits_stable_and_peak_limits());
    assert!(!projection.has_full_recovery_reserve());
}

#[test]
fn should_reject_a_soft_projection_overflow() {
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
        .expect_err("an overflowing soft recovery projection must fail closed");

    assert_eq!(
        projection_error,
        AdaptiveRamGrowthGuardError::MemoryProjectionOverflow
    );
}

#[test]
fn should_keep_prefill_and_decode_transient_high_water_values_independent() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");

    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        2_000,
        10_000,
        0,
    );
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        2_000,
        2_100,
        2_600,
        0,
    );

    assert_eq!(
        adaptive_ram_growth_guard
            .observed_transient_high_water_bytes(AdaptiveRamGrowthPhase::Prefill),
        8_000
    );
    assert_eq!(
        adaptive_ram_growth_guard
            .observed_transient_high_water_bytes(AdaptiveRamGrowthPhase::Decode),
        500
    );
}

#[test]
fn should_record_a_completed_zero_transient_prefill_observation() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");

    assert!(
        !adaptive_ram_growth_guard
            .has_completed_growth_observation(AdaptiveRamGrowthPhase::Prefill)
    );
    assert!(
        !adaptive_ram_growth_guard.has_completed_growth_observation(AdaptiveRamGrowthPhase::Decode)
    );

    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        2_000,
        2_000,
        2_000,
        0,
    );

    assert!(
        adaptive_ram_growth_guard.has_completed_growth_observation(AdaptiveRamGrowthPhase::Prefill),
        "a completed prefill must count as observed even when it used no transient bytes"
    );
    assert!(
        !adaptive_ram_growth_guard.has_completed_growth_observation(AdaptiveRamGrowthPhase::Decode),
        "prefill evidence must not mark decode as observed"
    );
    assert_eq!(
        adaptive_ram_growth_guard
            .observed_transient_high_water_bytes(AdaptiveRamGrowthPhase::Prefill),
        0
    );
}

#[test]
fn should_preserve_adaptive_high_water_observations_when_the_limit_changes() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_DECODE_CONTEXT,
        true,
        4_000,
        5_000,
        6_000,
        0,
    );

    adaptive_ram_growth_guard
        .update_active_memory_limit_bytes(8_000)
        .expect("a positive live limit should be accepted");

    assert_eq!(
        adaptive_ram_growth_guard
            .observed_transient_high_water_bytes(AdaptiveRamGrowthPhase::Decode),
        1_000
    );
    assert_eq!(
        adaptive_ram_growth_guard
            .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 6_000, 500, 0, 0)
            .expect("the updated guard should project growth")
            .active_memory_limit_bytes(),
        8_000
    );
    assert_eq!(
        adaptive_ram_growth_guard
            .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 6_000, 500, 0, 0)
            .expect("the updated guard should project growth")
            .allowed_active_memory_bytes(),
        8_080
    );
}

#[test]
fn should_project_exact_temporary_workspace_without_double_counting_learned_residual_growth() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(2_000)
        .expect("a positive active-memory limit should create a guard");
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        400,
        500,
        800,
        200,
    );

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(DEFAULT_PREFILL_CONTEXT, 500, 100, 0, 200)
        .expect("exact temporary and residual bytes should project without overflow");

    assert_eq!(projection.exact_temporary_workspace_bytes(), 200);
    assert_eq!(projection.observed_transient_high_water_bytes(), 100);
    assert_eq!(projection.stable_projected_bytes(), 600);
    assert_eq!(projection.peak_projected_bytes(), 900);
    assert_eq!(projection.soft_recovery_projected_bytes(), 1_200);
}

#[test]
fn should_reserve_a_routed_expert_page_alongside_lazy_persistent_growth_after_a_live_limit_reduction()
 {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(28_000_000_000)
        .expect("the reproduced live MLX limit should create a guard");

    let projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(1, false, true),
            27_806_577_158,
            192_061_440,
            70_778_880,
            0,
        )
        .expect("the reproduced paged decode projection should fit the platform range");

    assert_eq!(
        projection.routed_expert_page_reservation_bytes(),
        70_778_880
    );
    assert_eq!(projection.stable_projected_bytes(), 28_069_417_478);
    assert_eq!(projection.required_reclamation_bytes(), 69_417_478);
}
