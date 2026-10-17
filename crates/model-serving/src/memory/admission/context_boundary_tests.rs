//! Boundary contracts for context admission and retained-expert capacity.
//!
//! The Reclaim/Reject edges here are the exact decisions a long agent
//! conversation exercises on every turn: a projection one byte over the
//! ceiling must demand exactly that byte of elastic retention, never more,
//! and a projection that retention cannot cover must reject with the exact
//! shortfall instead of demanding a reclamation that cannot succeed.

use crate::{
    ContextAdmissionRequirements, MemoryAdmissionDecision, MemoryBoundary,
    VISIBLE_ANSWER_RESERVE_TOKENS, bounded_output_reservation_token_count,
    retained_expert_payload_capacity_bytes, safe_minimum_active_memory_ceiling_bytes,
};

fn requirements_for_current_state(
    current_active_memory_bytes: usize,
    retained_expert_payload_bytes: usize,
) -> ContextAdmissionRequirements {
    ContextAdmissionRequirements {
        current_active_memory_bytes,
        context_growth_bytes: 100,
        expert_page_reservation_bytes: 20,
        temporary_workspace_bytes: 30,
        retained_expert_payload_bytes,
        active_memory_ceiling_bytes: 1_000,
        complete_experts_are_resident: false,
    }
}

#[test]
fn should_admit_when_the_projection_lands_exactly_on_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        requirements_for_current_state(850, 400).decide(),
        MemoryAdmissionDecision::Admit,
        "850 + 100 + 20 + 30 = 1000 must be admissible at a 1000-byte ceiling"
    );
}

#[test]
fn should_demand_only_the_overflow_bytes_when_retention_covers_them() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        requirements_for_current_state(950, 100).decide(),
        MemoryAdmissionDecision::Reclaim {
            required_bytes: 100
        },
        "950 + 150 = 1100, one hundred over the ceiling with 100 elastic bytes retained must reclaim exactly 100"
    );
    assert_eq!(
        requirements_for_current_state(950, 150).decide(),
        MemoryAdmissionDecision::Reclaim {
            required_bytes: 100
        },
        "surplus retention must not inflate the demanded reclamation"
    );
}

#[test]
fn should_reject_with_the_exact_shortfall_when_retention_is_one_byte_short() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        requirements_for_current_state(950, 99).decide(),
        MemoryAdmissionDecision::Reject {
            boundary: MemoryBoundary::StableActiveCeiling,
            shortfall_bytes: 1,
        },
        "100 needed, 99 retained: the caller must learn it is exactly one byte short"
    );
}

#[test]
fn should_reject_with_the_full_overflow_when_no_elastic_retention_exists() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        requirements_for_current_state(950, 0).decide(),
        MemoryAdmissionDecision::Reject {
            boundary: MemoryBoundary::StableActiveCeiling,
            shortfall_bytes: 100,
        },
        "a dense or fully evicted worker must not be told to reclaim anything"
    );
}

#[test]
fn should_reject_with_the_maximum_shortfall_when_the_projection_overflows() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        requirements_for_current_state(usize::MAX, 1_000).decide(),
        MemoryAdmissionDecision::Reject {
            boundary: MemoryBoundary::StableActiveCeiling,
            shortfall_bytes: u64::MAX,
        },
        "an unrepresentable projection must fail closed without panicking"
    );
}

#[test]
fn should_admit_an_empty_request_against_a_zero_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        ContextAdmissionRequirements {
            current_active_memory_bytes: 0,
            context_growth_bytes: 0,
            expert_page_reservation_bytes: 0,
            temporary_workspace_bytes: 0,
            retained_expert_payload_bytes: 0,
            active_memory_ceiling_bytes: 0,
            complete_experts_are_resident: false,
        }
        .decide(),
        MemoryAdmissionDecision::Admit,
        "an all-zero observation is arithmetically consistent and must admit"
    );
}

#[test]
fn should_floor_the_safe_minimum_ceiling_at_zero_when_idle_memory_is_fully_evictable() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(500, 500, 60),
        60,
        "fully evictable idle memory leaves only the page reserve as the floor"
    );
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(200, 500, 60),
        60,
        "more evictable payload than idle memory must saturate at zero, not wrap"
    );
    assert_eq!(
        safe_minimum_active_memory_ceiling_bytes(500, 100, 60),
        460,
        "a partially evictable idle footprint keeps its non-evictable remainder"
    );
}

#[test]
fn should_bound_the_output_reservation_by_the_thinking_allowance_plus_the_answer_reserve() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        bounded_output_reservation_token_count(65_535, Some(16_384)),
        18_432,
        "the 2026-10-16 agent shape: 65,535 requested with a 16,384 thinking budget reserves thinking plus one answer chunk, not the full budget"
    );
    assert_eq!(VISIBLE_ANSWER_RESERVE_TOKENS, 2_048);
}

#[test]
fn should_never_reserve_more_output_than_the_caller_configured() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        bounded_output_reservation_token_count(512, Some(16_384)),
        512
    );
    assert_eq!(bounded_output_reservation_token_count(300, None), 300);
}

#[test]
fn should_reserve_the_visible_answer_allowance_when_no_thinking_budget_exists() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        bounded_output_reservation_token_count(65_535, None),
        2_048,
        "without a thinking budget only the visible-answer reserve is pre-held; longer answers grow under the per-step guard"
    );
}

#[test]
fn should_saturate_when_the_thinking_allowance_nears_the_word_size() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        bounded_output_reservation_token_count(usize::MAX, Some(usize::MAX)),
        usize::MAX
    );
}

#[test]
fn should_reserve_the_largest_page_when_no_expert_growth_is_pending() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let capacity_bytes = retained_expert_payload_capacity_bytes(1_000, 2_000, 100, 50, 0, 300, 0);
    assert_eq!(
        capacity_bytes, 1_200,
        "reserve = max(pending allocation, largest page) = 100; capacity = 300 retained + 900 slack"
    );
}

#[test]
fn should_charge_pending_experts_and_the_largest_page_together_when_retention_grows() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let capacity_bytes = retained_expert_payload_capacity_bytes(1_000, 2_000, 100, 50, 0, 300, 200);
    assert_eq!(
        capacity_bytes, 1_350,
        "reserve = 50 + 100 = 150; capacity = (300 + 200) + (2_000 - 1_150)"
    );
}

#[test]
fn should_shrink_retained_capacity_by_exactly_the_overage_when_reserves_exceed_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let capacity_bytes = retained_expert_payload_capacity_bytes(1_950, 2_000, 100, 50, 0, 300, 0);
    assert_eq!(
        capacity_bytes, 250,
        "live reserve 1_950 + 100 overflows the 2_000 ceiling by 50: capacity = 300 - 50"
    );
}

#[test]
fn should_zero_retained_capacity_when_future_reserves_overflow_the_word() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    assert_eq!(
        retained_expert_payload_capacity_bytes(1_000, 2_000, 100, u64::MAX, 0, 300, 200,),
        0,
        "an unrepresentable reserve must fail closed to zero capacity, not wrap large"
    );
}
