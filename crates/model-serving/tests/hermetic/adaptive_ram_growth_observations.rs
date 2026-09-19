//! Completed-observation and live-limit contracts for adaptive RAM growth.
//!
//! These tests are separate from projection admission so each test owner stays
//! readable: this file explains how completed forwards teach reusable transient
//! evidence, while `adaptive_ram_growth_guard.rs` covers projection boundaries.

use astronomical_model_serving::{AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, MemoryPhase};

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
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        8_000
    );
    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Decode),
        500
    );
}

#[test]
fn should_cap_decode_warming_with_decode_evidence_instead_of_the_all_phase_maximum() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");

    // One huge prefill teaches a large transient window; decode teaches a
    // small one. Issue #512: decode warming must reserve against decode's own
    // workspace, not stay suppressed by prefill's spent transient forever.
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

    let decode_ceiling_bytes = adaptive_ram_growth_guard.hot_expert_retention_ceiling_bytes(
        MemoryPhase::Decode,
        2_100,
        0,
        0,
    );
    let prefill_ceiling_bytes = adaptive_ram_growth_guard.hot_expert_retention_ceiling_bytes(
        MemoryPhase::Prefill,
        2_100,
        0,
        0,
    );

    // Ceiling = retained + (ceiling + 1% allowance - active - phase reserve).
    // Decode reserve is 500 bytes; the all-phase maximum is 8_000 bytes.
    assert_eq!(decode_ceiling_bytes, 10_000 + 100 - 2_100 - 500);
    assert_eq!(prefill_ceiling_bytes, 10_000 + 100 - 2_100 - 8_000);
    assert!(
        decode_ceiling_bytes > prefill_ceiling_bytes,
        "decode warming must claim more than a prefill-sized reserve allows"
    );
}

#[test]
fn should_fall_back_to_the_all_phase_maximum_before_the_phase_has_evidence() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");

    // Only prefill has observed anything. The first decode warming steps must
    // stay conservative until decode teaches its own workspace.
    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        2_000,
        10_000,
        0,
    );

    let decode_ceiling_bytes = adaptive_ram_growth_guard.hot_expert_retention_ceiling_bytes(
        MemoryPhase::Decode,
        2_000,
        0,
        0,
    );

    assert_eq!(decode_ceiling_bytes, 10_000 + 100 - 2_000 - 8_000);
}

#[test]
fn should_record_a_completed_zero_transient_prefill_observation() {
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(10_000)
        .expect("a positive active-memory limit should create a guard");

    assert!(!adaptive_ram_growth_guard.has_completed_growth_observation(MemoryPhase::Prefill));
    assert!(!adaptive_ram_growth_guard.has_completed_growth_observation(MemoryPhase::Decode));

    adaptive_ram_growth_guard.record_completed_growth_for_context(
        DEFAULT_PREFILL_CONTEXT,
        true,
        2_000,
        2_000,
        2_000,
        0,
    );

    assert!(
        adaptive_ram_growth_guard.has_completed_growth_observation(MemoryPhase::Prefill),
        "a completed prefill must count as observed even when it used no transient bytes"
    );
    assert!(
        !adaptive_ram_growth_guard.has_completed_growth_observation(MemoryPhase::Decode),
        "prefill evidence must not mark decode as observed"
    );
    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
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
        .update_active_memory_ceiling_bytes(8_000)
        .expect("a positive live limit should be accepted");

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Decode),
        1_000
    );
    assert_eq!(
        adaptive_ram_growth_guard
            .project_growth_for_context(DEFAULT_DECODE_CONTEXT, 6_000, 500, 0, 0)
            .expect("the updated guard should project growth")
            .active_memory_ceiling_bytes(),
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
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
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

#[test]
fn should_exclude_streamed_and_evicted_expert_pages_from_the_learned_prefill_transient() {
    // Issue #691: paged prefill promotes expert pages that appear in the MLX
    // peak but are evicted before completion (net resident delta zero). The
    // subtract chain must charge them to expert ownership, not activation:
    // baseline 1,000 stays stable, the forward peaks with 200 transient
    // activation bytes plus 400 streamed-and-evicted page bytes, workspace 0.
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(20_000)
        .expect("a positive active-memory limit should create a guard");

    adaptive_ram_growth_guard.record_completed_growth_excluding_expert_page_stream(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        1_000,
        1_600,
        0,
        0,
        400,
    );

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        200
    );
}

#[test]
fn should_exclude_only_the_stream_bytes_beyond_the_retained_expert_delta() {
    // One forward promotes 500 page bytes and retains 300 of them: the retained
    // delta already sits inside the post-forward active baseline, so only the
    // 200 evicted stream bytes leave the residual. peak = 1,300 stable
    // (1,000 + 300 retained) + 200 activation + 200 evicted stream.
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(20_000)
        .expect("a positive active-memory limit should create a guard");

    adaptive_ram_growth_guard.record_completed_growth_excluding_expert_page_stream(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        1_300,
        1_700,
        0,
        300,
        500,
    );

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        200
    );
}

#[test]
fn should_never_subtract_retained_pages_twice_from_the_learned_transient() {
    // Fully retained promotion with zero eviction: peak - stable already
    // excludes the retained pages through the post-forward active sample, so
    // the stream evidence must cost the residual nothing (issue #691 guard:
    // erasing real activation headroom would let the next forward overfill
    // retention). peak 1, earmark stable 1,000 + 400 retained = 1,400, peak =
    // 1,400 + 250 activation + 0 evicted.
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(20_000)
        .expect("a positive active-memory limit should create a guard");

    adaptive_ram_growth_guard.record_completed_growth_excluding_expert_page_stream(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        1_400,
        1_650,
        0,
        400,
        400,
    );

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        250
    );
}

#[test]
fn should_keep_the_older_learning_contract_unchanged_without_expert_page_stream_evidence() {
    // The pre-#691 method delegates to the same core with zero stream bytes, so
    // every existing observation shape is preserved verbatim (#623/#644 pins).
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

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        8_000
    );
}

#[test]
fn should_saturate_to_zero_when_stream_evidence_exceeds_the_whole_residual() {
    // Expert churn plus streaming can leave sampled stream bytes larger than
    // the observed peak window (allocator timing differences across events).
    // Under-projection is recoverable by design; the learned high-water must
    // clamp instead of wrapping.
    let mut adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(20_000)
        .expect("a positive active-memory limit should create a guard");

    adaptive_ram_growth_guard.record_completed_growth_excluding_expert_page_stream(
        DEFAULT_PREFILL_CONTEXT,
        true,
        1_000,
        1_000,
        1_300,
        100,
        0,
        900,
    );

    assert_eq!(
        adaptive_ram_growth_guard.observed_transient_high_water_bytes(MemoryPhase::Prefill),
        0
    );
}
