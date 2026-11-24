//! Re-sizing per-layer tables during a replay.

use std::collections::BTreeSet;

use super::rebalanced_replay::rebalanced_layer_slot_limits;
use super::{
    LeastRecentlyUsed, LimitedLookaheadBelady, ReplayCase, RoutedReply, replay,
    replay_with_rebalanced_tables,
};

const EXPERT_BYTES: u64 = 10;

/// Three layers: layer 0 cycles over four experts, layers 1 and 2
/// always route to the same one. Each layer starts with a two-slot table.
fn three_layer_case(token_count: u16) -> ReplayCase {
    let mut token_routes = Vec::new();
    for token_index in 0..token_count {
        token_routes.push(vec![vec![token_index % 4], vec![0], vec![0]]);
    }
    ReplayCase {
        routed_reply: RoutedReply::from_token_routes(8, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES; 3],
        capacity_bytes: 6 * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![Vec::new(); 3],
        layer_slot_limits: Some(vec![2, 2, 2]),
        pinned_layer_indexes: Vec::new(),
    }
}

fn window_of(expert_ids: &[u16]) -> BTreeSet<u16> {
    expert_ids.iter().copied().collect()
}

#[test]
fn should_give_a_broad_layer_more_slots_than_a_narrow_one() {
    let replay_case = three_layer_case(4);
    let windows = vec![window_of(&[0, 1, 2, 3]), window_of(&[0]), window_of(&[0])];

    let new_limits = rebalanced_layer_slot_limits(&replay_case, &windows, 1);

    assert_eq!(new_limits, vec![4, 1, 1]);
}

#[test]
fn should_keep_every_table_at_its_minimum_size() {
    let replay_case = three_layer_case(4);
    let windows = vec![
        window_of(&[0, 1, 2, 3, 4, 5]),
        window_of(&[]),
        window_of(&[]),
    ];

    let new_limits = rebalanced_layer_slot_limits(&replay_case, &windows, 2);

    assert!(new_limits.iter().all(|&slot_limit| slot_limit >= 2));
}

#[test]
fn should_leave_a_pinned_layer_its_size() {
    let replay_case = ReplayCase {
        pinned_layer_indexes: vec![1],
        layer_slot_limits: Some(vec![2, 8, 2]),
        ..three_layer_case(4)
    };
    let windows = vec![window_of(&[0, 1, 2]), window_of(&[0]), window_of(&[0])];

    let new_limits = rebalanced_layer_slot_limits(&replay_case, &windows, 1);

    assert_eq!(new_limits[1], 8);
    assert_eq!(new_limits[0] + new_limits[2], 4);
}

#[test]
fn should_match_a_fixed_replay_when_it_never_rebalances() {
    let replay_case = three_layer_case(24);

    let never_rebalanced =
        replay_with_rebalanced_tables(&replay_case, &mut LeastRecentlyUsed::default(), 0, 1);
    let fixed = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert_eq!(never_rebalanced.miss_count, fixed.miss_count);
}

#[test]
fn should_miss_less_once_slots_move_to_the_broad_layer() {
    let replay_case = three_layer_case(48);

    let rebalanced =
        replay_with_rebalanced_tables(&replay_case, &mut LeastRecentlyUsed::default(), 6, 1);
    let fixed = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert!(rebalanced.miss_count < fixed.miss_count);
}

#[test]
fn should_treat_a_lookahead_past_the_reply_end_as_the_whole_reply() {
    let replay_case = three_layer_case(12);

    let huge_lookahead =
        replay(&replay_case, &mut LimitedLookaheadBelady::new(usize::MAX)).miss_count;
    let whole_reply = replay(&replay_case, &mut LimitedLookaheadBelady::new(12)).miss_count;

    assert_eq!(huge_lookahead, whole_reply);
}
