//! Boundary tests for the context-admission workspace accounting: overflow
//! arms, the seated-peak identity, and the safe-minimum ceiling after
//! reclamation.

use crate::admission::context::{
    ContextAdmissionRequirements, combined_persistent_growth_bytes,
    persistent_context_restore_workspace_bytes, safe_minimum_active_memory_ceiling_bytes,
    seated_complete_expert_request_peak_active_memory_bytes,
    seated_complete_expert_request_temporary_workspace_bytes,
};
use crate::vocabulary::{MemoryAdmissionDecision, MemoryBoundary};

#[test]
fn should_reject_context_admission_when_the_projection_overflows() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: usize::MAX,
        context_growth_bytes: 1,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: 0,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: usize::MAX,
        complete_experts_are_resident: false,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryAdmissionDecision::Reject {
            boundary: MemoryBoundary::StableActiveCeiling,
            shortfall_bytes: u64::MAX,
        }
    );
}

#[test]
fn should_admit_context_growth_landing_exactly_on_the_ceiling() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: 90,
        context_growth_bytes: 5,
        expert_page_reservation_bytes: 4,
        temporary_workspace_bytes: 1,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: 100,
        complete_experts_are_resident: false,
    }
    .decide();

    assert_eq!(decision, MemoryAdmissionDecision::Admit);
}

#[test]
fn should_demand_exact_reclamation_one_byte_past_the_ceiling() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: 90,
        context_growth_bytes: 5,
        expert_page_reservation_bytes: 4,
        temporary_workspace_bytes: 2,
        retained_expert_payload_bytes: 10,
        active_memory_ceiling_bytes: 100,
        complete_experts_are_resident: false,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryAdmissionDecision::Reclaim { required_bytes: 1 }
    );
}

#[test]
fn should_reclaim_up_to_the_exact_retention_boundary() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: 90,
        context_growth_bytes: 5,
        expert_page_reservation_bytes: 4,
        temporary_workspace_bytes: 2,
        retained_expert_payload_bytes: 1,
        active_memory_ceiling_bytes: 100,
        complete_experts_are_resident: false,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryAdmissionDecision::Reclaim { required_bytes: 1 },
        "a one-byte deficit against one retained byte sits exactly on the \
         retention boundary and reclaims rather than rejects"
    );
}

#[test]
fn should_reject_with_the_shortfall_beyond_retention_when_growth_exceeds_both() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: 90,
        context_growth_bytes: 5,
        expert_page_reservation_bytes: 4,
        temporary_workspace_bytes: 2,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: 100,
        complete_experts_are_resident: false,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryAdmissionDecision::Reject {
            boundary: MemoryBoundary::StableActiveCeiling,
            shortfall_bytes: 1,
        }
    );
}

#[test]
fn should_prefer_complete_residency_demotion_over_reclamation_when_experts_are_resident() {
    let decision = ContextAdmissionRequirements {
        current_active_memory_bytes: 90,
        context_growth_bytes: 5,
        expert_page_reservation_bytes: 4,
        temporary_workspace_bytes: 2,
        retained_expert_payload_bytes: 100,
        active_memory_ceiling_bytes: 100,
        complete_experts_are_resident: true,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryAdmissionDecision::DemoteCompleteResidency {
            reassess_after_demotion: true,
        },
        "an indivisible complete owner must demote before any byte-level \
         reclamation is ordered"
    );
}

#[test]
fn should_make_the_seated_temporary_workspace_reproduce_the_seated_peak_exactly() {
    // The documented identity: current + context_growth + temporary equals
    // the seated peak across every ordering of restore overlap and context
    // growth, including both boundary crossings.
    for (context_growth_bytes, restore_overlap_workspace_bytes) in
        [(100_usize, 40_usize), (40, 100), (75, 75), (0, 0)]
    {
        let publication_workspace_bytes = 7_usize;
        let current_active_memory_bytes = 1_000_usize;
        let seated_peak_active_memory_bytes =
            seated_complete_expert_request_peak_active_memory_bytes(
                current_active_memory_bytes,
                context_growth_bytes,
                restore_overlap_workspace_bytes,
                publication_workspace_bytes,
            )
            .expect("bounded inputs must not overflow");
        let temporary_workspace_bytes = seated_complete_expert_request_temporary_workspace_bytes(
            context_growth_bytes,
            restore_overlap_workspace_bytes,
            publication_workspace_bytes,
        )
        .expect("bounded inputs must not overflow");

        assert_eq!(
            current_active_memory_bytes + context_growth_bytes + temporary_workspace_bytes,
            seated_peak_active_memory_bytes,
            "the workspace identity must hold for context {context_growth_bytes} \
             and restore {restore_overlap_workspace_bytes}"
        );
    }
}

#[test]
fn should_return_none_from_every_workspace_combiner_that_overflows() {
    assert!(persistent_context_restore_workspace_bytes(usize::MAX, 2).is_none());
    assert!(seated_complete_expert_request_peak_active_memory_bytes(usize::MAX, 0, 0, 1).is_none());
    assert!(combined_persistent_growth_bytes(usize::MAX, 1).is_none());
}

#[test]
fn should_floor_the_safe_minimum_ceiling_at_the_page_reserve_when_eviction_exceeds_idle() {
    // Reclaiming more than the idle footprint cannot go below zero, but the
    // pager still needs its one page reserve above the floor.
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(100, 700, 25),
        25,
        "eviction beyond the idle footprint saturates at zero before the \
         page reserve is added"
    );
    assert_eq!(safe_minimum_active_memory_ceiling_bytes(100, 70, 25), 55);
    assert_eq!(safe_minimum_active_memory_ceiling_bytes(0, 0, 0), 0);
}
