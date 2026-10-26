//! Boundary contracts for the request-stable prefill residency contract.
//!
//! The contract freezes pinning at request open and may only shrink. These
//! tests pin the shrink and re-admission edges: over-demanded reclamation
//! demotes every pin, exhausted headroom must never re-seat a streamed layer
//! (the #1278 thrash loop), and generation phases must discard the contract
//! instead of replaying prefill decisions during decode.

use crate::phase::MemoryPhase;
use crate::residency::ExpertLayerGeometry;
use crate::residency::{
    CurrentExpertLayerResidency, ExpertLayerResidencyTarget, ExpertResidencyPlan,
    RequestExpertLayerRole, RequestExpertResidency, RetainedExpertPageClass,
    publish_request_stable_residency_plan,
    retained_complete_layer_ceiling_after_prefill_budget_refresh,
    retained_resident_ceiling_after_budget_refresh, seated_complete_layer_count,
};

pub(super) fn uniform_geometry(layer_count: usize) -> Vec<ExpertLayerGeometry> {
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

#[test]
fn should_demote_every_pinned_layer_when_the_reclamation_exceeds_all_pins() {
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

#[test]
fn should_leave_the_contract_unchanged_when_nothing_must_yield() {
    let geometries = uniform_geometry(3);
    let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0, 1], 80));

    let unchanged = request_residency.shrink_after_capacity_failure(0, &geometries);

    assert_eq!(
        unchanged.pinned_complete_layer_indexes(),
        vec![0, 1],
        "a zero-byte reclamation must not touch the pin set"
    );
}

#[test]
fn should_never_reseat_a_streamed_layer_when_pinned_claims_exhaust_the_ceiling() {
    let geometries = uniform_geometry(3);
    // Pins claim 80 of an 80-byte ceiling, leaving zero headroom for layer 2.
    let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0, 1], 80));

    let stabilized =
        request_residency.stabilize_prefill_plan(candidate_plan(Vec::new(), 80), &[], &geometries);

    assert_eq!(
        stabilized.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "a streamed layer the exhausted budget cannot hold must stream, not re-seat"
    );
    assert_eq!(stabilized.maximum_new_retained_bytes, 0);
    assert_eq!(stabilized.complete_layer_targets, vec![0, 1]);
}

#[test]
fn should_admit_a_streamed_layer_at_the_exact_headroom_boundary_only() {
    let geometries = uniform_geometry(3);
    // One pin claims 40 of 80, leaving exactly one 40-byte complete layer of headroom.
    let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0], 80));

    let stabilized =
        request_residency.stabilize_prefill_plan(candidate_plan(Vec::new(), 80), &[], &geometries);

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

#[test]
fn should_floor_a_refreshed_ceiling_at_payload_already_resident() {
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

#[test]
fn should_discard_the_prefill_contract_for_generation_phases() {
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
}

#[test]
fn should_shrink_but_never_expand_an_open_prefill_contract() {
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
}

fn streamed_target_count(plan: &ExpertResidencyPlan) -> usize {
    plan.layer_targets
        .iter()
        .filter(|target| matches!(target, ExpertLayerResidencyTarget::StreamOperationLocal))
        .count()
}

#[test]
fn should_keep_authorizing_new_retention_until_every_streamed_layer_is_seated() {
    // The 2026-10-24 flight: a multi-chunk request's retention allowance read
    // zero on every chunk after the first while two thirds of the paid ceiling
    // sat unused, because nothing pinned the allowance to the unseated target.
    // A mixed contract (one pin, two streamed layers) over a ceiling that fits
    // all three layers must authorize new retention on every chunk until the
    // last streamed layer seats, and only then converge to zero.
    let geometries = uniform_geometry(3);
    let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0], 120));

    // Chunk 1: the pin is seated; both streamed layers fit the leftover 80.
    let seated_pin = CurrentExpertLayerResidency {
        layer_index: 0,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    };
    let chunk_one = request_residency.stabilize_prefill_plan(
        candidate_plan(Vec::new(), 120),
        std::slice::from_ref(&seated_pin),
        &geometries,
    );
    assert_eq!(
        chunk_one.layer_targets[1],
        ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
    );
    assert_eq!(
        chunk_one.layer_targets[2],
        ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
    );
    assert_eq!(
        chunk_one.maximum_new_retained_bytes, 80,
        "both unseated streamed layers must be covered by the retention allowance"
    );

    // Chunk 2: one more layer seated. The seated streamed layer is now
    // preserved like a seated pin — its payload stops re-consuming stream
    // headroom — so the allowance covers exactly the one remaining unseated
    // streamed layer.
    let seated_pin_and_first_streamed = vec![
        seated_pin,
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];
    let chunk_two = request_residency.stabilize_prefill_plan(
        candidate_plan(Vec::new(), 120),
        &seated_pin_and_first_streamed,
        &geometries,
    );
    assert_eq!(
        chunk_two.layer_targets[1],
        ExpertLayerResidencyTarget::PreserveComplete,
        "a seated streamed layer must be preserved, not re-charged to the ceiling"
    );
    assert_eq!(
        chunk_two.layer_targets[2],
        ExpertLayerResidencyTarget::AdmitCompleteOnMandatoryRead
    );
    assert_eq!(
        chunk_two.maximum_new_retained_bytes, 40,
        "the allowance must shrink to exactly the unseated streamed layer"
    );

    // Chunk 3: the whole target is seated. Every streamed layer is preserved,
    // nothing streams, and the allowance is zero because nothing is unseated.
    let fully_seated = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];
    let chunk_three = request_residency.stabilize_prefill_plan(
        candidate_plan(Vec::new(), 120),
        &fully_seated,
        &geometries,
    );
    assert_eq!(streamed_target_count(&chunk_three), 0);
    assert_eq!(chunk_three.maximum_new_retained_bytes, 0);
    assert_eq!(
        chunk_three.expected_preserved_bytes, 120,
        "every seated layer — pin or streamed — counts in preserved bytes"
    );
}

#[test]
fn should_count_only_actually_resident_complete_layers_as_seated() {
    // The 2026-10-24 flight logged pinned_complete_layer_count=34 beside
    // current_residency_count=9: the contract's pin count (intent) was
    // published as if it were occupancy. The seated count must come from the
    // topology's complete-layer classification, never from the contract.
    let current_residencies = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1],
            payload_bytes: 20,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 2,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];

    assert_eq!(
        seated_complete_layer_count(&current_residencies),
        2,
        "partial (elastic) residency must never count toward seated complete layers"
    );
    assert_eq!(seated_complete_layer_count(&[]), 0);
}

#[test]
fn should_count_only_seated_complete_pins_in_expected_preserved_bytes() {
    // expected_preserved_bytes feeds the utilization telemetry, so it must
    // count exactly the complete-layer payload preserved by the plan. Warm
    // partial tables (elastic routed experts) are retention too, but they are
    // not complete layers and must not inflate this number.
    let geometries = uniform_geometry(3);
    let request_residency = RequestExpertResidency::open_prefill(&candidate_plan(vec![0], 120));
    let mixed_residency = vec![
        CurrentExpertLayerResidency {
            layer_index: 0,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::ElasticRoutedExperts,
            retained_expert_ids: vec![0, 1],
            payload_bytes: 20,
            covered_weighted_demand: 0,
        },
    ];

    let stabilized = request_residency.stabilize_prefill_plan(
        candidate_plan(Vec::new(), 120),
        &mixed_residency,
        &geometries,
    );

    assert_eq!(
        stabilized.expected_preserved_bytes, 40,
        "a 20-byte partial warm table beside a 40-byte complete layer must not \
         be counted in expected_preserved_bytes"
    );
}

#[test]
fn should_never_grow_the_streamed_layer_count_across_chunks_of_one_request() {
    // With a constant budget, seating progress must never plan MORE streamed
    // layers than the previous chunk planned. The count is structurally
    // stable — a seated streamed layer keeps its idempotent adopt-on-read
    // offer and keeps consuming its own payload from the same ceiling — but
    // growth chunk over chunk would be the re-stream spiral the utilization
    // journey exists to catch, so the direction is pinned here.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0], 80));

    let seated_pin = CurrentExpertLayerResidency {
        layer_index: 0,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    };
    let chunk_one = request_residency.stabilize_prefill_plan(
        four_layer_candidate(Vec::new(), 80),
        std::slice::from_ref(&seated_pin),
        &geometries,
    );

    let seated_pin_and_one = vec![
        seated_pin,
        CurrentExpertLayerResidency {
            layer_index: 1,
            class: RetainedExpertPageClass::StableCompleteLayer,
            retained_expert_ids: vec![0, 1, 2, 3],
            payload_bytes: 40,
            covered_weighted_demand: 0,
        },
    ];
    let chunk_two = request_residency.stabilize_prefill_plan(
        four_layer_candidate(Vec::new(), 80),
        &seated_pin_and_one,
        &geometries,
    );

    let chunk_one_streamed_count = streamed_target_count(&chunk_one);
    let chunk_two_streamed_count = streamed_target_count(&chunk_two);
    assert_eq!(chunk_one_streamed_count, 2);
    assert!(
        chunk_two_streamed_count <= chunk_one_streamed_count,
        "seating progress must never plan more streamed layers than the \
         previous chunk: {chunk_two_streamed_count} > {chunk_one_streamed_count}"
    );
}

pub(super) fn four_layer_candidate(
    complete_layer_targets: Vec<usize>,
    ceiling_bytes: u64,
) -> ExpertResidencyPlan {
    ExpertResidencyPlan {
        phase: MemoryPhase::Prefill,
        retained_expert_ceiling_bytes: ceiling_bytes,
        complete_layer_targets,
        layer_targets: vec![ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead; 4],
        reserved_routed_overlay_bytes: 0,
        expected_preserved_bytes: 0,
        maximum_new_retained_bytes: 0,
        deterministic_release_order: Vec::new(),
        is_low_budget_partial_mode: false,
    }
}

#[test]
fn should_not_claim_pinned_layers_beyond_the_composed_ceiling() {
    // The 2026-10-23 under-utilization flight: 34 pins claimed
    // 29,091,692,544 bytes against a 28,938,405,632-byte ceiling. Pins
    // frozen at request open must not keep claiming past a ceiling that
    // context growth shrank: the over-claim saturates stream-admission
    // headroom to zero while naming promotion targets the budget cannot
    // honor. Three pins of 40 against a 100-byte ceiling: the first two
    // claim 80; the third pin's claim does not fit and must fall through to
    // streamed handling instead of promising a promotion it cannot fund.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0, 1, 2], 160));

    let stabilized =
        request_residency.stabilize_prefill_plan(candidate_plan(Vec::new(), 100), &[], &geometries);

    assert_eq!(
        stabilized.layer_targets[0],
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
    );
    assert_eq!(
        stabilized.layer_targets[1],
        ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead
    );
    assert_eq!(
        stabilized.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "a pin whose claim cannot fit the composed ceiling must stream, not \
         promise an unfunded promotion"
    );
    assert_eq!(stabilized.complete_layer_targets, vec![0, 1]);
    assert_eq!(stabilized.maximum_new_retained_bytes, 0);
}

#[test]
fn should_keep_a_seated_pinned_layer_claimed_even_when_unseated_pins_cannot_fit() {
    // A seated pin's bytes are already resident: its claim always holds, and
    // only the unseated pins compete for whatever ceiling remains.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0, 1, 2], 160));
    let seated_layer = CurrentExpertLayerResidency {
        layer_index: 0,
        class: RetainedExpertPageClass::StableCompleteLayer,
        retained_expert_ids: vec![0, 1, 2, 3],
        payload_bytes: 40,
        covered_weighted_demand: 0,
    };

    let stabilized = request_residency.stabilize_prefill_plan(
        candidate_plan(Vec::new(), 60),
        &[seated_layer],
        &geometries,
    );

    assert_eq!(
        stabilized.layer_targets[0],
        ExpertLayerResidencyTarget::PreserveComplete,
        "the seated pin keeps its claim regardless of the shrunken ceiling"
    );
    assert_eq!(stabilized.expected_preserved_bytes, 40);
    assert_eq!(
        stabilized.layer_targets[1],
        ExpertLayerResidencyTarget::StreamOperationLocal,
        "unseated pins must fit within the ceiling left after seated claims"
    );
    assert_eq!(stabilized.complete_layer_targets, vec![0]);
}

#[test]
fn should_demote_only_the_tail_pin_when_one_claim_misses_the_ceiling() {
    // Boundary: three pins claim exactly 120 of 120. All fit; none demotes.
    let geometries = uniform_geometry(4);
    let request_residency =
        RequestExpertResidency::open_prefill(&four_layer_candidate(vec![0, 1, 2], 120));

    let exact_fit =
        request_residency.stabilize_prefill_plan(candidate_plan(Vec::new(), 120), &[], &geometries);
    assert_eq!(exact_fit.complete_layer_targets, vec![0, 1, 2]);
    assert_eq!(
        exact_fit.layer_targets[3],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );

    // One byte less and only the tail pin yields, mirroring the contract's
    // tail-first shrink order.
    let one_byte_short =
        request_residency.stabilize_prefill_plan(candidate_plan(Vec::new(), 119), &[], &geometries);
    assert_eq!(one_byte_short.complete_layer_targets, vec![0, 1]);
    assert_eq!(
        one_byte_short.layer_targets[2],
        ExpertLayerResidencyTarget::StreamOperationLocal
    );
}

#[test]
fn should_keep_planner_targets_beyond_the_contract_span_instead_of_truncating() {
    // A candidate plan longer than the opened contract must never be
    // truncated down to the contract's span: layers beyond the roles keep
    // the planner's disposition for them. Truncating dropped those layers'
    // targets entirely, leaving callers to index out of bounds.
    let geometries = uniform_geometry(3);
    let mut one_layer_opening = candidate_plan(vec![0], 80);
    one_layer_opening.layer_targets =
        vec![ExpertLayerResidencyTarget::PromoteCompleteOnMandatoryRead];
    let request_residency = RequestExpertResidency::open_prefill(&one_layer_opening);
    let mut longer_candidate = candidate_plan(Vec::new(), 80);
    longer_candidate.layer_targets =
        vec![ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead; 3];

    let stabilized = request_residency.stabilize_prefill_plan(longer_candidate, &[], &geometries);

    assert_eq!(stabilized.layer_targets.len(), 3);
    assert_eq!(
        stabilized.layer_targets[2],
        ExpertLayerResidencyTarget::AdmitPartialOnMandatoryRouteRead,
        "the planner target beyond the contract span must survive stabilization"
    );
}
