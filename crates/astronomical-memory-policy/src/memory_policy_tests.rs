use crate::admission::allocation::{AllocationAdmissionDecision, AllocationAdmissionObservation};
use crate::admission::complete_residency::{
    CompleteResidencyDecision, CompleteResidencyRequirements,
};
use crate::ceiling::{MemoryCeilingChangeDecision, MemoryCeilingChangeRequirements};
use crate::reclamation::complete_residency_exceeds_ceiling_with_activation_headroom;
use crate::residency::classify_expert_memory_mode;
use crate::vocabulary::MemoryBoundary;
use astronomical_ipc_protocol::ExpertMemoryMode;

#[test]
fn should_identify_the_allocation_boundary_and_required_shortfall() {
    let observation = AllocationAdmissionObservation::new(900, 0, 125, 1_000);

    assert_eq!(
        observation.decide(),
        AllocationAdmissionDecision::Reject {
            boundary: MemoryBoundary::AllocationProjection,
            shortfall_bytes: 25,
        }
    );
}

#[test]
fn should_plan_demotion_before_installing_a_lower_ceiling() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 25_000,
        requested_ceiling_bytes: 23_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 24_000,
        retained_paged_expert_payload_bytes: 20_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: true,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_demote_complete_residency_when_only_the_idle_snapshot_fits() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 30_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: true,
            retained_paged_expert_reclamation_bytes: 0,
        }
    );
}

#[test]
fn should_keep_complete_residency_when_idle_memory_and_serving_headroom_fit() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 32_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: 2_000,
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
fn should_preserve_paged_reclamation_when_complete_residency_headroom_is_irrelevant() {
    let decision = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 38_000,
        requested_ceiling_bytes: 30_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 33_000,
        retained_paged_expert_payload_bytes: 2_000,
        complete_experts_are_resident: false,
        complete_residency_required_headroom_bytes: u64::MAX,
    }
    .decide();

    assert_eq!(
        decision,
        MemoryCeilingChangeDecision::Lower {
            must_demote_complete_residency: false,
            retained_paged_expert_reclamation_bytes: 2_000,
        }
    );
}

#[test]
fn should_not_request_demotion_for_raised_or_unchanged_ceilings() {
    let requirements = MemoryCeilingChangeRequirements {
        current_ceiling_bytes: 30_000,
        requested_ceiling_bytes: 38_000,
        minimum_safe_ceiling_bytes: 10_000,
        current_active_memory_bytes: 29_000,
        retained_paged_expert_payload_bytes: 25_000,
        complete_experts_are_resident: true,
        complete_residency_required_headroom_bytes: u64::MAX,
    };

    assert_eq!(
        requirements.decide(),
        MemoryCeilingChangeDecision::Raise {
            may_attempt_complete_residency: true,
        }
    );
    assert_eq!(
        MemoryCeilingChangeRequirements {
            requested_ceiling_bytes: requirements.current_ceiling_bytes,
            ..requirements
        }
        .decide(),
        MemoryCeilingChangeDecision::Unchanged
    );
}

#[test]
fn should_request_allocator_cleanup_without_confusing_cache_with_active_memory() {
    let observation = AllocationAdmissionObservation::new(800, 300, 100, 1_000);

    assert_eq!(
        observation.decide(),
        AllocationAdmissionDecision::ClearMlxBufferCacheThenAdmit
    );
}

#[test]
fn should_decide_complete_residency_from_one_replacement_aware_policy() {
    let requirements = CompleteResidencyRequirements {
        current_active_memory_bytes: 2_000,
        retained_paged_expert_payload_bytes: 600,
        complete_expert_payload_bytes: 1_000,
        required_headroom_bytes: 500,
        active_memory_ceiling_bytes: 3_000,
    };

    assert_eq!(
        requirements.decide(),
        CompleteResidencyDecision::Admit {
            projected_active_memory_bytes: 2_400,
            required_headroom_bytes: 500,
        }
    );
}

#[test]
fn should_admit_complete_residency_without_prefill_three_layer_weight_headroom() {
    const PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES: u64 = 28_193_105_290;
    const ACTIVE_MEMORY_CEILING_BYTES: u64 = 33_000_000_000;
    const GATE_UP_FUSION_TRANSIENT_PAYLOAD_BYTES: u64 = 1_073_741_824;
    const PREFILL_THREE_LAYER_WEIGHT_HEADROOM_BYTES: u64 = 4_831_838_208;

    assert!(
        !complete_residency_exceeds_ceiling_with_activation_headroom(
            PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES,
            ACTIVE_MEMORY_CEILING_BYTES,
            GATE_UP_FUSION_TRANSIENT_PAYLOAD_BYTES,
        )
    );
    assert!(complete_residency_exceeds_ceiling_with_activation_headroom(
        PROJECTED_RESIDENT_ACTIVE_MEMORY_BYTES,
        ACTIVE_MEMORY_CEILING_BYTES,
        PREFILL_THREE_LAYER_WEIGHT_HEADROOM_BYTES,
    ));
}

#[test]
fn should_classify_complete_owner_as_resident_and_empty_pager_cache_as_paged() {
    assert_eq!(
        classify_expert_memory_mode(true, true, 0),
        ExpertMemoryMode::Resident
    );
    assert_eq!(
        classify_expert_memory_mode(false, false, 0),
        ExpertMemoryMode::Resident
    );
    assert_eq!(
        classify_expert_memory_mode(false, true, 0),
        ExpertMemoryMode::Paged
    );
    assert_eq!(
        classify_expert_memory_mode(false, true, 15_854_469_120),
        ExpertMemoryMode::Hybrid
    );
}
