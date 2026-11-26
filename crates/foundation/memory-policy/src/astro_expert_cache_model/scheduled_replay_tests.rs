//! A recorded reply replayed under the capacity it really had.

use super::{
    CapacityTrim, LeastRecentlyUsed, ReplayCase, RoutedReply, replay,
    replay_under_capacity_schedule,
};

const EXPERT_BYTES: u64 = 10;

/// Two layers, each token routing expert 0 in both, with room for three
/// experts: layer 0 holds two at the start and layer 1 one.
fn two_layer_case(token_count: usize) -> ReplayCase {
    let mut token_routes = Vec::with_capacity(token_count);
    for _ in 0..token_count {
        token_routes.push(vec![vec![0], vec![0]]);
    }
    ReplayCase {
        routed_reply: RoutedReply::from_token_routes(4, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES, EXPERT_BYTES],
        capacity_bytes: 3 * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![vec![0, 1], vec![0]],
        layer_slot_limits: None,
        pinned_layer_indexes: Vec::new(),
    }
}

#[test]
fn should_match_the_fixed_capacity_replay_when_the_schedule_never_moves() {
    let replay_case = two_layer_case(6);
    let fixed = replay(&replay_case, &mut LeastRecentlyUsed::default());
    let scheduled = replay_under_capacity_schedule(
        &replay_case,
        &[3 * EXPERT_BYTES; 6],
        &mut LeastRecentlyUsed::default(),
        CapacityTrim::DropLeastOccupiedLayer,
    );
    assert_eq!(scheduled.replay_outcome, fixed);
    assert_eq!(scheduled.dropped_layer_count, 0);
}

#[test]
fn should_drop_the_least_occupied_layer_whole_for_a_one_expert_dip() {
    let replay_case = two_layer_case(4);
    let dip = [30, 20, 30, 30];
    let scheduled = replay_under_capacity_schedule(
        &replay_case,
        &dip,
        &mut LeastRecentlyUsed::default(),
        CapacityTrim::DropLeastOccupiedLayer,
    );
    // Layer 1 holds one expert, the fewest: it goes, and is read back once.
    assert_eq!(scheduled.dropped_layer_count, 1);
    assert_eq!(scheduled.trimmed_expert_count, 1);
    assert_eq!(scheduled.replay_outcome.miss_count, 1);
}

#[test]
fn should_give_up_only_what_the_dip_needs_expert_by_expert() {
    let mut replay_case = two_layer_case(4);
    replay_case.held_expert_ids_by_layer_at_start = vec![vec![0, 1], vec![0, 1]];
    replay_case.capacity_bytes = 4 * EXPERT_BYTES;
    let dip = [40, 30, 40, 40];
    let by_layer = replay_under_capacity_schedule(
        &replay_case,
        &dip,
        &mut LeastRecentlyUsed::default(),
        CapacityTrim::DropLeastOccupiedLayer,
    );
    let by_expert = replay_under_capacity_schedule(
        &replay_case,
        &dip,
        &mut LeastRecentlyUsed::default(),
        CapacityTrim::GiveUpLowestPriorityExpert,
    );
    assert_eq!(
        by_layer.trimmed_expert_count, 2,
        "a whole two-expert layer goes"
    );
    assert_eq!(by_expert.trimmed_expert_count, 1);
    assert_eq!(by_expert.dropped_layer_count, 0);
    assert!(by_expert.replay_outcome.miss_count <= by_layer.replay_outcome.miss_count);
}

#[test]
fn should_give_up_everything_when_the_capacity_falls_to_zero() {
    let replay_case = two_layer_case(2);
    let scheduled = replay_under_capacity_schedule(
        &replay_case,
        &[3 * EXPERT_BYTES, 0],
        &mut LeastRecentlyUsed::default(),
        CapacityTrim::DropLeastOccupiedLayer,
    );
    assert_eq!(scheduled.trimmed_expert_count, 3);
    assert_eq!(scheduled.dropped_layer_count, 2);
    assert_eq!(
        scheduled.replay_outcome.hit_count, 2,
        "only the first token, served before the fall, finds its experts"
    );
}
