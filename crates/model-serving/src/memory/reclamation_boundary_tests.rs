//! Boundary contracts for expert reclamation under pressure.
//!
//! These tests pin the edges where a reclamation demand can be provably
//! unsatisfiable: when even zero retained experts cannot fit the fixed
//! forward, when the retry target is zero, and when repeated reclaim passes
//! must terminate in Reject instead of looping forever against retention
//! that can no longer cover the deficit.

use astronomical_test_harness::offline_denied_test;

use crate::{
    ExpertReclamationPlan, PagedExpertReclamationStep,
    expert_reclamation_bytes_to_fit_fixed_forward,
    fixed_forward_workspace_after_allocation_failure, next_paged_expert_reclamation_step,
    should_retry_fixed_forward_after_expert_reclamation,
};

offline_denied_test!(
    should_demand_the_full_retained_payload_when_even_zero_experts_cannot_fit,
    {
        // non-elastic active = 30 - 25 = 5; ceiling 32 leaves 27 for experts plus
        // workspace, but the fixed forward needs 30. Even a fully evicted worker
        // cannot satisfy this forward; the formula's answer is "reclaim all 25",
        // and the caller must reject rather than evict-and-still-fail.
        let required_reclamation_bytes =
            expert_reclamation_bytes_to_fit_fixed_forward(30, 25, 32, 30);

        assert_eq!(
            required_reclamation_bytes, 25,
            "an unsatisfiable forward must demand only the full retained payload, never more"
        );
    }
);

offline_denied_test!(
    should_return_the_excess_experts_only_when_the_forward_can_actually_fit,
    {
        // non-elastic 5; ceiling 32 - 5 - 20 = 7 keepable; 25 - 7 = 18 demanded.
        let required_reclamation_bytes =
            expert_reclamation_bytes_to_fit_fixed_forward(30, 25, 32, 20);

        assert_eq!(required_reclamation_bytes, 18);
    }
);

offline_denied_test!(
    should_keep_the_observed_high_water_when_it_exceeds_the_failed_forward_shape,
    {
        let workspace_bytes =
            fixed_forward_workspace_after_allocation_failure(36_000, 37_000, 500, 2_000);

        assert_eq!(
            workspace_bytes, 2_000,
            "a completed forward's high-water remains the retry lower bound"
        );
    }
);

offline_denied_test!(
    should_never_authorize_a_retry_when_the_reclamation_target_is_zero,
    {
        assert!(
            !should_retry_fixed_forward_after_expert_reclamation(false, 1_000, 0, 0),
            "a zero-byte reclamation target means eviction can change nothing; retrying repeats the failure"
        );
    }
);

offline_denied_test!(
    should_reject_immediately_when_retention_can_never_cover_the_deficit,
    {
        let unsatisfiable_plan =
            ExpertReclamationPlan::for_projected_memory(130, 130, 130, 100, 120, 5);

        assert_eq!(unsatisfiable_plan.required_reclamation_bytes(), 30);
        assert_eq!(unsatisfiable_plan.reclamation_target_bytes(), 5);
        assert_eq!(unsatisfiable_plan.unresolved_shortfall_bytes(), 25);
        assert!(!unsatisfiable_plan.can_satisfy_every_memory_boundary());
        assert_eq!(
            next_paged_expert_reclamation_step(false, unsatisfiable_plan, None),
            PagedExpertReclamationStep::Reject,
            "a deficit retention cannot cover must reject before any eviction pass runs"
        );
    }
);

offline_denied_test!(
    should_fail_closed_when_no_deficit_exists_yet_the_caller_reports_a_miss,
    {
        let fitting_plan = ExpertReclamationPlan::for_projected_memory(90, 110, 110, 100, 120, 50);

        assert_eq!(fitting_plan.required_reclamation_bytes(), 0);
        assert_eq!(
            next_paged_expert_reclamation_step(false, fitting_plan, None),
            PagedExpertReclamationStep::Reject,
            "every projection fits but the caller still reports a miss: eviction has nothing to fix"
        );
    }
);

offline_denied_test!(
    should_take_a_single_max_of_equal_deficits_rather_than_summing,
    {
        let plan = ExpertReclamationPlan::for_projected_memory(110, 120, 105, 100, 120, 50);

        assert_eq!(
            plan.required_reclamation_bytes(),
            10,
            "equal stable and peak deficits of 10 must reclaim 10 once, not 20"
        );
    }
);

offline_denied_test!(
    should_terminate_the_reclaim_loop_in_reject_once_retention_runs_out,
    {
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
            let plan = ExpertReclamationPlan::for_projected_memory(
                130,
                130,
                130,
                100,
                120,
                usize::try_from(retained_expert_payload_bytes).expect("payload fits usize"),
            );
            match next_paged_expert_reclamation_step(false, plan, Some(true)) {
                PagedExpertReclamationStep::Reclaim { target_bytes } => {
                    retained_expert_payload_bytes =
                        retained_expert_payload_bytes.saturating_sub(target_bytes as u64);
                }
                terminal_step => break terminal_step,
            }
        };

        assert_eq!(decision, PagedExpertReclamationStep::Reject);
        assert_eq!(
            retained_expert_payload_bytes, 20,
            "the final 20 retained bytes cannot cover the 30-byte deficit and must survive, not be churned"
        );
        assert_eq!(pass_count, 2);
    }
);
