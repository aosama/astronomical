//! Unit tests for assigning states to the parts of a training.

use super::*;

/// `state_count` distinct keys spread far apart, as hashes are.
fn spread_keys(state_count: u64) -> Vec<u64> {
    let mut keys = Vec::new();
    for state_number in 0..state_count {
        keys.push(state_number.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    }
    keys
}

fn count_of(parts: &HashMap<u64, SplitPart>, wanted_part: SplitPart) -> usize {
    let mut count = 0;
    for part in parts.values() {
        count += usize::from(*part == wanted_part);
    }
    count
}

#[test]
fn should_divide_forty_states_fifteen_fifteen_seventy() {
    let parts = split_parts_by_state(&spread_keys(40));
    assert_eq!(count_of(&parts, SplitPart::Judging), 6);
    assert_eq!(count_of(&parts, SplitPart::Choosing), 6);
    assert_eq!(count_of(&parts, SplitPart::Training), 28);
}

#[test]
fn should_give_every_repeat_of_a_state_one_part() {
    let mut keys = spread_keys(25);
    keys.extend(spread_keys(25));
    let parts = split_parts_by_state(&keys);
    assert_eq!(parts.len(), 25);
}

#[test]
fn should_split_the_same_states_the_same_way_in_any_order() {
    let mut reversed = spread_keys(33);
    reversed.reverse();
    assert_eq!(
        split_parts_by_state(&spread_keys(33)),
        split_parts_by_state(&reversed)
    );
}

#[test]
fn should_assign_nothing_without_states() {
    assert!(split_parts_by_state(&[]).is_empty());
}

#[test]
fn should_train_on_every_state_below_one_whole_cycle() {
    let parts = split_parts_by_state(&spread_keys(19));
    assert_eq!(count_of(&parts, SplitPart::Training), 19);
}

#[test]
fn should_hold_states_back_from_one_whole_cycle() {
    let parts = split_parts_by_state(&spread_keys(20));
    assert_eq!(count_of(&parts, SplitPart::Judging), 3);
    assert_eq!(count_of(&parts, SplitPart::Choosing), 3);
    assert_eq!(count_of(&parts, SplitPart::Training), 14);
}

#[test]
fn should_train_a_single_state() {
    let parts = split_parts_by_state(&spread_keys(1));
    assert_eq!(parts.len(), 1);
    assert_eq!(count_of(&parts, SplitPart::Training), 1);
}
