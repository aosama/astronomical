//! Policy numbers for decode-warm fill after a pressure demotion.
//!
//! These tests do not load a model. They lock the two rules a later reader
//! must not "simplify":
//!
//! 1. After request pressure, decode-warm still spends the leftover composed
//!    budget (tens of gigabytes), not a 1 GB routed working set.
//! 2. Complete resident experts are one atomic owner, so native prefill
//!    recovery must demote that owner before it can reclaim expert bytes.

use astronomical_model_serving::{
    prefill_recovery_must_demote_complete_resident_owner, retained_expert_fill_budget_bytes,
};

#[test]
fn should_keep_full_decode_warm_fill_budget_when_the_model_was_not_forced_paged() {
    // A quiet idle model still gets the leftover composed plan, not a
    // smaller "just in case" working set.
    let planned_retained_expert_budget_bytes: u64 = 30_133_404_494;
    let requested_retained_expert_payload_bytes = u64::MAX;

    let retained_expert_fill_budget_bytes = retained_expert_fill_budget_bytes(
        planned_retained_expert_budget_bytes,
        requested_retained_expert_payload_bytes,
    );

    assert_eq!(
        retained_expert_fill_budget_bytes,
        planned_retained_expert_budget_bytes
    );
}

#[test]
fn should_use_the_composed_decode_warm_fill_budget_after_request_pressure() {
    // These bytes came from a real 25 GB Ornith 4-bit log: leftover composed
    // decode budget was ~24.67 GB, complete experts were ~25.77 GB, and one
    // routed page times 40 layers was only ~1.07 GB. Decode must spend the
    // leftover, not the 1.07 GB working set.
    let planned_retained_expert_budget_bytes: u64 = 24_672_184_486;
    let complete_expert_payload_bytes: u64 = 25_769_803_776;
    let decoder_layer_count: u64 = 40;
    let largest_routed_expert_page_bytes: u64 = 26_738_688;
    let decode_working_set_budget_bytes =
        decoder_layer_count.saturating_mul(largest_routed_expert_page_bytes);

    let retained_expert_fill_budget_bytes =
        retained_expert_fill_budget_bytes(planned_retained_expert_budget_bytes, u64::MAX);

    assert_eq!(
        retained_expert_fill_budget_bytes,
        planned_retained_expert_budget_bytes
    );
    assert!(retained_expert_fill_budget_bytes < complete_expert_payload_bytes);
    assert!(
        retained_expert_fill_budget_bytes > decode_working_set_budget_bytes,
        "decode must reclaim the leftover composed budget instead of staying on a 1 GB working set"
    );
}

#[test]
fn should_honor_an_explicit_smaller_requested_fill_cap() {
    // A diagnostic caller may still ask for less than the composed leftover.
    // Decode handoff does not do this; it passes u64::MAX.
    let largest_routed_expert_page_bytes: u64 = 26_738_688;

    let retained_expert_fill_budget_bytes =
        retained_expert_fill_budget_bytes(30_133_404_494, largest_routed_expert_page_bytes);

    assert_eq!(
        retained_expert_fill_budget_bytes,
        largest_routed_expert_page_bytes
    );
}

#[test]
fn should_demote_a_complete_resident_owner_during_native_prefill_recovery() {
    assert!(prefill_recovery_must_demote_complete_resident_owner(true));
    assert!(!prefill_recovery_must_demote_complete_resident_owner(false));
}
