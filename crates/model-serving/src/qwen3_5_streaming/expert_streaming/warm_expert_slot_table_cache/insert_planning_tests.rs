//! Beside-source coverage for the pure slot-planning unit of the warm-table
//! insert: the free-first, eviction-fill plan assembly. The growth and write
//! paths run under the serial engine lane against real arrays.

use super::assemble_planned_slots;

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
