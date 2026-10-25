//! Boundary, negative, and happy-path tests for the live ceiling change
//! decision, including the exact-arrival boundaries on every threshold.

use crate::ceiling::MemoryCeilingChangeDecision;
use crate::ceiling::MemoryCeilingChangeRequirements;
use crate::vocabulary::MemoryBoundary;

#[test]
fn should_reject_a_ceiling_below_the_minimum_safe_bound() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 79,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 10,
        retained_paged_expert_payload_bytes: 0,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Reject {
            boundary: MemoryBoundary::LiveCeilingMinimum,
            shortfall_bytes: 1,
        }
    );
}

#[test]
fn should_accept_a_raise_arriving_exactly_at_the_minimum_safe_bound() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 60,
        requested_ceiling_bytes: 80,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 10,
        retained_paged_expert_payload_bytes: 0,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Raise {
            may_attempt_complete_residency: true,
        }
    );
}

#[test]
fn should_return_unchanged_when_the_requested_ceiling_matches_current() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 90,
        requested_ceiling_bytes: 90,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 10,
        retained_paged_expert_payload_bytes: 0,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert_eq!(decision, MemoryCeilingChangeDecision::Unchanged);
}

#[test]
fn should_reject_an_unchanged_request_when_current_already_sits_below_the_minimum() {
    // The minimum bound is checked before the no-op: keeping a ceiling that
    // violates safe ownership is still a rejection, never a silent pass.
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 70,
        requested_ceiling_bytes: 70,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 10,
        retained_paged_expert_payload_bytes: 0,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert!(matches!(
        decision,
        MemoryCeilingChangeDecision::Reject {
            shortfall_bytes: 10,
            ..
        }
    ));
}

#[test]
fn should_lower_without_reclamation_when_active_memory_already_fits() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 90,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 50,
        retained_paged_expert_payload_bytes: 40,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_reclaim_exactly_the_active_deficit_when_it_is_below_the_retained_payload() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 70,
        minimum_safe_ceiling_bytes: 60,
        current_active_memory_bytes: 100,
        retained_paged_expert_payload_bytes: 200,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 30,
        }
    );
}

#[test]
fn should_clamp_reclamation_to_the_retained_paged_payload_at_and_beyond_it() {
    let requirements_at_the_boundary = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 50,
        minimum_safe_ceiling_bytes: 40,
        current_active_memory_bytes: 100,
        retained_paged_expert_payload_bytes: 50,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: 0,
    };
    assert_eq!(
        requirements_at_the_boundary.decide(),
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 50,
        },
        "a deficit exactly equal to the retained payload reclaims all of it"
    );

    let requirements_beyond_the_boundary = MemoryCeilingChangeRequirements {
        requested_ceiling_bytes: 40,
        current_active_memory_bytes: 100,
        ..requirements_at_the_boundary
    };
    assert_eq!(
        requirements_beyond_the_boundary.decide(),
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 50,
        },
        "a deficit beyond the retained payload still reclaims only what exists"
    );
}

#[test]
fn should_keep_complete_ownership_when_headroom_fits_exactly_at_the_lower_ceiling() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 90,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 60,
        retained_paged_expert_payload_bytes: 20,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 30,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 0,
        },
        "active 60 plus headroom 30 lands exactly on 90, so no demotion is ordered"
    );
}

#[test]
fn should_demand_complete_demotion_one_byte_of_headroom_past_the_lower_ceiling() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 100,
        requested_ceiling_bytes: 90,
        minimum_safe_ceiling_bytes: 80,
        current_active_memory_bytes: 60,
        retained_paged_expert_payload_bytes: 20,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 31,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: true,
            retained_paged_expert_reclamation_bytes: 0,
        },
        "while complete experts are resident the deficit routes through demotion, not page reclamation"
    );
}
