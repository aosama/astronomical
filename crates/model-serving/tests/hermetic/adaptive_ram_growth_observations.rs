//! Completed-observation and live-limit contracts for adaptive RAM growth.
//!
//! These tests are separate from projection admission so each test owner stays
//! readable: this file explains how completed forwards teach reusable transient
//! evidence, while `adaptive_ram_growth_guard.rs` covers projection boundaries.

use astronomical_model_serving::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthPhase,
};

const DEFAULT_DECODE_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::decode(1, false, false);
const DEFAULT_PREFILL_CONTEXT: AdaptiveRamGrowthContext =
    AdaptiveRamGrowthContext::prefill(128, 0, false, false, false);

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
    assert_eq!(projection.recovery_projected_bytes(), 1_200);
    assert_eq!(
        projection.forward_reserve_bytes(),
        400,
        "the residency planner must reserve every byte between the admitted active baseline and expected peak boundary"
    );
}

#[test]
fn should_not_subtract_stable_expert_growth_twice_from_prefill_headroom() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(40_000)
        .expect("a positive active-memory limit should create a guard");

    // Active memory grows by 10,000 stable expert bytes. Peak is another 3,000
    // bytes above the final active sample, so the reusable transient window is
    // exactly 3,000 bytes. The stable post-forward sample already excludes the
    // expert growth from this difference.
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        20_000,
        30_000,
        33_000,
        0,
    );

    assert_eq!(
        adaptive_ram_growth_guard
            .observed_transient_high_water_bytes(AdaptiveRamGrowthPhase::Prefill),
        3_000
    );
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
    assert_eq!(
        projection.operation_reclamation_required_bytes(),
        69_417_478
    );
}
