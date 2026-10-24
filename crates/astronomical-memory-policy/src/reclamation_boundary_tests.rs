//! Boundary contracts for expert reclamation under pressure.
//!
//! These tests pin the edges where a reclamation demand can be provably
//! unsatisfiable: when even zero retained experts cannot fit the fixed
//! forward, when the retry target is zero, and when repeated reclaim passes
//! must terminate in Reject instead of looping forever against retention
//! that can no longer cover the deficit.

use crate::budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
};
use crate::forward_admission::PagedForwardAdmissionDecision;
use crate::reclamation::{
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure,
    should_retry_fixed_forward_after_expert_reclamation,
};

fn paged_forward_projection(
    stable_projected_bytes: usize,
    peak_projected_bytes: usize,
    active_memory_ceiling_bytes: usize,
) -> crate::budget::AdaptiveRamGrowthProjection {
    AdaptiveRamGrowthGuard::new(active_memory_ceiling_bytes)
        .expect("valid ceiling")
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(1, AdaptiveRamGrowthExecutionProfile::Paged),
            stable_projected_bytes,
            0,
            0,
            0,
            peak_projected_bytes.saturating_sub(stable_projected_bytes),
        )
        .expect("the projection should fit the platform range")
}

#[test]
fn should_demand_the_full_retained_payload_when_even_zero_experts_cannot_fit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // non-elastic active = 30 - 25 = 5; ceiling 32 leaves 27 for experts plus
    // workspace, but the fixed forward needs 30. Even a fully evicted worker
    // cannot satisfy this forward; the formula's answer is "reclaim all 25",
    // and the caller must reject rather than evict-and-still-fail.
    let required_reclamation_bytes = expert_reclamation_bytes_to_fit_fixed_forward(30, 25, 32, 30);

    assert_eq!(
        required_reclamation_bytes, 25,
        "an unsatisfiable forward must demand only the full retained payload, never more"
    );
}

#[test]
fn should_return_the_excess_experts_only_when_the_forward_can_actually_fit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // non-elastic 5; ceiling 32 - 5 - 20 = 7 keepable; 25 - 7 = 18 demanded.
    let required_reclamation_bytes = expert_reclamation_bytes_to_fit_fixed_forward(30, 25, 32, 20);

    assert_eq!(required_reclamation_bytes, 18);
}

#[test]
fn should_keep_the_observed_high_water_when_it_exceeds_the_failed_forward_shape() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let workspace_bytes =
        fixed_forward_workspace_after_allocation_failure(36_000, 37_000, 500, 2_000);

    assert_eq!(
        workspace_bytes, 2_000,
        "a completed forward's high-water remains the retry lower bound"
    );
}

#[test]
fn should_never_authorize_a_retry_when_the_reclamation_target_is_zero() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert!(
        !should_retry_fixed_forward_after_expert_reclamation(false, 1_000, 0, 0),
        "a zero-byte reclamation target means eviction can change nothing; retrying repeats the failure"
    );
}

#[test]
fn should_reject_immediately_when_retention_can_never_cover_the_deficit() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let unsatisfiable_projection = paged_forward_projection(130, 130, 100);
    let decision =
        PagedForwardAdmissionDecision::for_paged_forward(&unsatisfiable_projection, 5, None);

    assert_eq!(
        decision,
        PagedForwardAdmissionDecision::Reject {
            unresolved_shortfall_bytes: 25
        },
        "a deficit retention cannot cover must reject before any eviction pass runs"
    );
}

#[test]
fn should_admit_when_the_projection_fits_even_after_a_prior_reclaim_pass() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let fitting_projection = paged_forward_projection(90, 100, 100);

    assert_eq!(
        PagedForwardAdmissionDecision::for_paged_forward(&fitting_projection, 50, Some(false)),
        PagedForwardAdmissionDecision::Admit,
        "the projection, rather than a stale caller boolean, owns the decision"
    );
}

#[test]
fn should_take_a_single_max_of_equal_deficits_rather_than_summing() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let forward_projection = paged_forward_projection(110, 110, 100);

    assert_eq!(
        forward_projection.operation_reclamation_required_bytes(),
        10,
        "equal stable and peak deficits of 10 must reclaim 10 once, not 20"
    );
}

#[test]
fn should_terminate_the_reclaim_loop_in_reject_once_retention_runs_out() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    // Worst-case loop shape: each pass releases its full target but the MLX
    // active sample never drops (an immutable execution snapshot still holds
    // the evicted arrays). The deficit stays 30 while retention shrinks:
    // pass 1 reclaims 30 of 50; pass 2 sees a 30-byte deficit against only
    // 20 retained bytes and must Reject BEFORE evicting them, because even
    // evicting everything cannot fit the forward.
    let mut retained_expert_payload_bytes = 50_u64;
    let mut pass_count = 0_u32;
    let decision = loop {
        pass_count += 1;
        assert!(
            pass_count <= 4,
            "the reclaim loop must terminate within a few passes"
        );
        let forward_projection = paged_forward_projection(130, 130, 100);
        match PagedForwardAdmissionDecision::for_paged_forward(
            &forward_projection,
            usize::try_from(retained_expert_payload_bytes).expect("payload fits usize"),
            Some(true),
        ) {
            PagedForwardAdmissionDecision::ReclaimRetainedExperts { target_bytes } => {
                retained_expert_payload_bytes =
                    retained_expert_payload_bytes.saturating_sub(target_bytes as u64);
            }
            terminal_step => break terminal_step,
        }
    };

    assert_eq!(
        decision,
        PagedForwardAdmissionDecision::Reject {
            unresolved_shortfall_bytes: 10
        }
    );
    assert_eq!(
        retained_expert_payload_bytes, 20,
        "the final 20 retained bytes cannot cover the 30-byte deficit and must survive, not be churned"
    );
    assert_eq!(pass_count, 2);
}
