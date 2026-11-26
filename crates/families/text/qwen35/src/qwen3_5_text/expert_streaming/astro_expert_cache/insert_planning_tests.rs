//! Beside-source coverage for the pure slot-planning unit of the warm-table
//! insert: the free-first, eviction-fill plan assembly. The growth and write
//! paths run under the serial engine lane against real arrays.

use super::super::decode_astro_expert_cache_policy::DecodeExpertSlotState;
use super::{assemble_planned_slots, rank_and_cap_candidate_victims};

#[test]
fn should_fill_the_plan_from_free_slots_in_order() {
    let planned_slots = assemble_planned_slots(&[3, 1, 7], &[], 2);

    assert_eq!(planned_slots, vec![3, 1]);
}

#[test]
fn should_fill_the_plan_to_the_exact_count_from_eviction_candidates() {
    // One free slot plus two victims seat three experts; the third victim
    // stays unplanned.
    let planned_slots = assemble_planned_slots(&[5], &[9, 2, 6], 3);

    assert_eq!(planned_slots, vec![5, 9, 2]);
}

#[test]
fn should_plan_nothing_for_an_empty_incoming_set() {
    let planned_slots = assemble_planned_slots(&[5, 9], &[2, 6], 0);

    assert!(planned_slots.is_empty());
}

/// Slot states whose experts were last read at `last_read_epochs`, each with
/// one read, so by epoch 10,000 every score has decayed to nothing and only
/// the last read tells them apart.
fn decayed_slot_states(last_read_epochs: &[u64]) -> Vec<DecodeExpertSlotState> {
    let mut slot_states = Vec::new();
    for &last_read_epoch in last_read_epochs {
        let mut slot_state = DecodeExpertSlotState::default();
        slot_state.record_decode_reuse(last_read_epoch);
        slot_states.push(slot_state);
    }
    slot_states
}

#[test]
fn should_cap_the_victims_after_ranking_them_not_before() {
    // Slots 0 and 1 were read most recently; 2 and 3 longest ago. A cap of
    // two must keep 3 and 2, whatever their slot numbers.
    let slot_states = decayed_slot_states(&[90, 80, 20, 10]);
    let mut evictable_slots = vec![0, 1, 2, 3];

    rank_and_cap_candidate_victims(&mut evictable_slots, &slot_states, 10_000, 1, 2);

    assert_eq!(evictable_slots, vec![3, 2]);
}

#[test]
fn should_keep_every_victim_under_a_cap_larger_than_their_count() {
    let slot_states = decayed_slot_states(&[5, 1]);
    let mut evictable_slots = vec![0, 1];

    rank_and_cap_candidate_victims(&mut evictable_slots, &slot_states, 10_000, 1, 8);

    assert_eq!(evictable_slots, vec![1, 0]);
}

#[test]
fn should_keep_no_victim_under_a_cap_of_zero() {
    let slot_states = decayed_slot_states(&[5, 1]);
    let mut evictable_slots = vec![0, 1];

    rank_and_cap_candidate_victims(&mut evictable_slots, &slot_states, 10_000, 1, 0);

    assert!(evictable_slots.is_empty());
}
