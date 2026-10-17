//! Boundary contracts for the request-stable prefill residency contract.
//!
//! The contract freezes pinning at request open and may only shrink. These
//! tests pin the shrink and re-admission edges: over-demanded reclamation
//! demotes every pin, exhausted headroom must never re-seat a streamed layer
//! (the #1278 thrash loop), and generation phases must discard the contract
//! instead of replaying prefill decisions during decode.

use astronomical_test_harness::offline_denied_test;

use crate::memory::residency::ExpertLayerGeometry;
use crate::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, ExpertResidencyPlan, MemoryPhase,
    RequestExpertLayerRole, RequestExpertResidency, RetainedExpertPageClass,
    publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    retained_resident_ceiling_after_budget_refresh,
};

fn uniform_geometry(layer_count: usize) -> Vec<ExpertLayerGeometry> {
    (0..layer_count)
        .map(|layer_index| ExpertLayerGeometry {
            layer_index,
            complete_layer_payload_bytes: 40,
            expert_payload_bytes: 10,
            expert_capacity: 4,
            experts_per_token: 2,
        })
        .collect()
}

fn candidate_plan(complete_layer_targets: Vec<usize>, ceiling_bytes: u64) -> ExpertResidencyPlan {
    ExpertResidencyPlan {
        phase: MemoryPhase::Prefill,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets,
        layer_targets: vec![ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead; 3],
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: 0,
        maximum_new_retained_bytes: 0,
        deterministic_release_order: Vec::new(),
        is_low_budget_partial_mode: false,
    }
}

offline_denied_test!(
    should_demote_every_pinned_layer_when_the_reclamation_exceeds_all_pins,
    {
        let geometries = uniform_geometry(3);
        let request_residency =
            RequestExpertResidency::open_prefill(&candidate_plan(vec![0, 1, 2], 120));

        let shrunk = request_residency.shrink_after_capacity_failure(200, &geometries);

        assert_eq!(
            shrunk.layer_role(0),
            Some(RequestExpertLayerRole::DemotedByCapacityFailure),
            "an unmeetable reclamation must demote every pin rather than half of them"
        );
        assert_eq!(
            shrunk.layer_role(1),
            Some(RequestExpertLayerRole::DemotedByCapacityFailure)
        );
        assert_eq!(
            shrunk.layer_role(2),
            Some(RequestExpertLayerRole::DemotedByCapacityFailure)
        );
        assert!(shrunk.pinned_complete_layer_indexes().is_empty());
    }
);

offline_denied_test!(
    should_leave_the_contract_unchanged_when_nothing_must_yield,
    {
        let geometries = uniform_geometry(3);
        let request_residency =
            RequestExpertResidency::open_prefill(&candidate_plan(vec![0, 1], 80));

        let unchanged = request_residency.shrink_after_capacity_failure(0, &geometries);

        assert_eq!(
            unchanged.pinned_complete_layer_indexes(),
            vec![0, 1],
            "a zero-byte reclamation must not touch the pin set"
        );
    }
);

offline_denied_test!(
    should_never_reseat_a_streamed_layer_when_pinned_claims_exhaust_the_ceiling,
    {
        let geometries = uniform_geometry(3);
        // Pins claim 80 of an 80-byte ceiling, leaving zero headroom for layer 2.
        let request_residency =
            RequestExpertResidency::open_prefill(&candidate_plan(vec![0, 1], 80));

        let stabilized = request_residency.stabilize_prefill_plan(
            candidate_plan(Vec::new(), 80),
            &[],
            &geometries,
        );

        assert_eq!(
            stabilized.layer_targets[2],
            ExpertLayerResidencyTarget::StreamOperationLocal,
            "a streamed layer the exhausted budget cannot hold must stream, not re-seat"
        );
        assert_eq!(stabilized.maximum_new_retained_bytes, 0);
        assert_eq!(stabilized.complete_layer_targets, vec![0, 1]);
    }
);

offline_denied_test!(
    should_admit_a_streamed_layer_at_the_exact_headroom_boundary_only,
    {
        let geometries = uniform_geometry(3);
        // One pin claims 40 of 80, leaving exactly one 40-byte complete layer of headroom.
        let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0], 80));

        let stabilized = request_residency.stabilize_prefill_plan(
            candidate_plan(Vec::new(), 80),
            &[],
            &geometries,
        );

        assert_eq!(
            stabilized.layer_targets[1],
            ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead,
            "a streamed layer must adopt its already-paid complete read at the exact boundary"
        );
        assert_eq!(stabilized.maximum_new_retained_bytes, 40);
        assert_eq!(
            stabilized.layer_targets[2],
            ExpertLayerResidencyTarget::StreamOperationLocal,
            "the next streamed layer must find zero headroom left and stream"
        );
    }
);

offline_denied_test!(
    should_floor_a_refreshed_ceiling_at_payload_already_resident,
    {
        assert_eq!(
            retained_complete_layer_ceiling_after_prefill_budget_refresh(30, 40),
            40,
            "a learned budget that tightens below seated payload must not force eviction"
        );
        assert_eq!(
            retained_complete_layer_ceiling_after_prefill_budget_refresh(40, 40),
            40
        );
        assert_eq!(
            retained_complete_layer_ceiling_after_prefill_budget_refresh(50, 40),
            50
        );
        assert_eq!(retained_resident_ceiling_after_budget_refresh(30, 45), 45);
        assert_eq!(retained_resident_ceiling_after_budget_refresh(45, 45), 45);
        assert_eq!(retained_resident_ceiling_after_budget_refresh(60, 45), 60);
    }
);

offline_denied_test!(should_discard_the_prefill_contract_for_generation_phases, {
    let geometries = uniform_geometry(3);
    let candidate = candidate_plan(vec![0], 80);

    let (request_residency, published_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Decode,
        None,
        candidate_plan(vec![0], 80),
        &[],
        40,
        &geometries,
    );

    assert!(
        request_residency.is_none(),
        "decode must not inherit prefill pins"
    );
    assert_eq!(
        published_plan, candidate,
        "the generation candidate must pass through unchanged"
    );
});

offline_denied_test!(should_shrink_but_never_expand_an_open_prefill_contract, {
    let geometries = uniform_geometry(3);

    let (opened_residency, _) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        None,
        candidate_plan(vec![0, 1], 80),
        &[],
        0,
        &geometries,
    );
    assert_eq!(
        opened_residency
            .as_ref()
            .expect("prefill must open a contract")
            .pinned_complete_layer_indexes(),
        vec![0, 1]
    );

    // A 40-byte release during the same request demotes only the tail pin.
    let (shrunk_residency, shrunk_plan) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        opened_residency.as_ref(),
        candidate_plan(vec![0, 1], 80),
        &[CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        }],
        40,
        &geometries,
    );
    assert_eq!(
        shrunk_residency
            .as_ref()
            .expect("prefill keeps its contract")
            .pinned_complete_layer_indexes(),
        vec![0],
        "the released 40 bytes must demote exactly the tail pin"
    );
    assert_eq!(
        shrunk_plan.layer_targets[1],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "a demoted layer must stream for the rest of the request"
    );
    assert_eq!(
        shrunk_plan.layer_targets[0],
        ExpertLayerResidencyTarget::PreserveComplete,
        "the surviving seated pin must be preserved"
    );

    // A later publish with no further release keeps the shrunk contract.
    let (stable_residency, _) = publish_request_stable_residency_plan(
        MemoryPhase::Prefill,
        shrunk_residency.as_ref(),
        candidate_plan(vec![0, 1], 80),
        &[],
        0,
        &geometries,
    );
    assert_eq!(
        stable_residency
            .as_ref()
            .expect("prefill keeps its contract")
            .pinned_complete_layer_indexes(),
        vec![0],
        "a zero-release republish must not re-pin the demoted layer"
    );
});
