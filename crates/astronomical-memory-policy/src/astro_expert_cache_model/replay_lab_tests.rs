//! The replay lab against traces whose answers are known.

use std::collections::BTreeSet;

use super::{
    BeladyMinimum, DecayedFrequency, FrequencyAdmittedRecency, KeepingRule, LeastFrequentlyUsed,
    LeastRecentlyUsed, ReplayCase, RoutedReply, replay,
};

const EXPERT_BYTES: u64 = 10;

/// A one-layer reply routing one expert per token, with room for
/// `capacity_experts` experts and nothing held at the start.
fn single_layer_case(expert_sequence: &[u16], capacity_experts: u64) -> ReplayCase {
    let token_routes: Vec<Vec<Vec<u16>>> = expert_sequence
        .iter()
        .map(|&expert_id| vec![vec![expert_id]])
        .collect();
    ReplayCase {
        routed_reply: RoutedReply::from_token_routes(8, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES],
        capacity_bytes: capacity_experts * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![Vec::new()],
        layer_slot_limits: None,
        pinned_layer_indexes: Vec::new(),
    }
}

fn every_rule() -> Vec<Box<dyn KeepingRule>> {
    vec![
        Box::new(LeastRecentlyUsed::default()),
        Box::new(LeastFrequentlyUsed::default()),
        Box::new(DecayedFrequency::new(4.0)),
        Box::new(FrequencyAdmittedRecency::default()),
        Box::new(BeladyMinimum::default()),
    ]
}

#[test]
fn should_count_the_misses_belady_and_recency_make_on_a_worked_trace() {
    // Room for two. Belady keeps 0 and 1 and reads 2 and 3 once each without
    // keeping them; recency keeps evicting what comes back next.
    let replay_case = single_layer_case(&[0, 1, 2, 0, 1, 3, 0, 1], 2);

    let belady_outcome = replay(&replay_case, &mut BeladyMinimum::default());
    let recency_outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert_eq!(belady_outcome.miss_count, 4);
    assert_eq!(belady_outcome.hit_count, 4);
    assert_eq!(recency_outcome.miss_count, 8);
    assert_eq!(belady_outcome.read_bytes, 4 * EXPERT_BYTES);
    assert!((belady_outcome.misses_per_token() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn should_match_an_exhaustive_search_for_the_fewest_misses() {
    let traces: [&[u16]; 6] = [
        &[0, 1, 2, 0, 1, 3, 0, 1],
        &[0, 1, 2, 3, 0, 1, 2, 3, 0],
        &[3, 3, 1, 2, 1, 3, 0, 2, 1, 0],
        &[0, 1, 0, 2, 0, 3, 0, 1, 2, 3],
        &[2, 1, 0, 2, 1, 0, 3, 3, 2, 1],
        &[0, 0, 0, 1, 1, 2],
    ];
    for expert_sequence in traces {
        for capacity_experts in 1..=3 {
            let replay_case = single_layer_case(expert_sequence, capacity_experts);
            let belady_outcome = replay(&replay_case, &mut BeladyMinimum::default());
            let fewest_misses = fewest_possible_misses(
                expert_sequence,
                capacity_experts as usize,
                &BTreeSet::new(),
            );
            assert_eq!(
                belady_outcome.miss_count, fewest_misses,
                "trace {expert_sequence:?} with room for {capacity_experts}"
            );
        }
    }
}

/// Every keep-or-skip and every eviction tried: the true minimum.
fn fewest_possible_misses(
    expert_sequence: &[u16],
    capacity_experts: usize,
    held_experts: &BTreeSet<u16>,
) -> u64 {
    let Some((&expert_id, later_experts)) = expert_sequence.split_first() else {
        return 0;
    };
    if held_experts.contains(&expert_id) {
        return fewest_possible_misses(later_experts, capacity_experts, held_experts);
    }
    let mut fewest_later_misses =
        fewest_possible_misses(later_experts, capacity_experts, held_experts);
    if held_experts.len() < capacity_experts {
        let mut with_newcomer = held_experts.clone();
        with_newcomer.insert(expert_id);
        fewest_later_misses = fewest_later_misses.min(fewest_possible_misses(
            later_experts,
            capacity_experts,
            &with_newcomer,
        ));
    } else {
        for &victim in held_experts {
            let mut with_newcomer = held_experts.clone();
            with_newcomer.remove(&victim);
            with_newcomer.insert(expert_id);
            fewest_later_misses = fewest_later_misses.min(fewest_possible_misses(
                later_experts,
                capacity_experts,
                &with_newcomer,
            ));
        }
    }
    1 + fewest_later_misses
}

#[test]
fn should_never_let_a_rule_beat_belady_on_a_recorded_reply() {
    let routed_reply = RoutedReply::from_recording_bytes(include_bytes!(
        "fixtures/shakespeare_summary_40_layers.routes"
    ))
    .expect("the fixture decodes");
    let layer_count = routed_reply.layer_count();
    let replay_case = ReplayCase {
        routed_reply,
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES; layer_count],
        capacity_bytes: 3_000 * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![Vec::new(); layer_count],
        layer_slot_limits: None,
        pinned_layer_indexes: Vec::new(),
    };
    let belady_misses = replay(&replay_case, &mut BeladyMinimum::default()).miss_count;
    for mut keeping_rule in every_rule() {
        let rule_misses = replay(&replay_case, keeping_rule.as_mut()).miss_count;
        assert!(
            belady_misses <= rule_misses,
            "{} missed {rule_misses}, Belady {belady_misses}",
            keeping_rule.name()
        );
    }
}

#[test]
fn should_count_nothing_for_an_empty_reply() {
    let replay_case = single_layer_case(&[], 2);

    for mut keeping_rule in every_rule() {
        let outcome = replay(&replay_case, keeping_rule.as_mut());
        assert_eq!(outcome.miss_count, 0);
        assert_eq!(outcome.hit_count, 0);
        assert!(outcome.misses_per_token().abs() < f64::EPSILON);
    }
}

#[test]
fn should_miss_every_use_when_nothing_can_be_kept() {
    let replay_case = single_layer_case(&[0, 0, 1, 1], 0);

    for mut keeping_rule in every_rule() {
        assert_eq!(replay(&replay_case, keeping_rule.as_mut()).miss_count, 4);
    }
}

#[test]
fn should_seat_only_as_many_starting_experts_as_the_capacity_holds() {
    let mut replay_case = single_layer_case(&[0, 1, 2], 2);
    replay_case.held_expert_ids_by_layer_at_start = vec![vec![0, 1, 2]];

    let outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    // Experts 0 and 1 fit and are hits; 2 did not fit and is read.
    assert_eq!(outcome.hit_count, 2);
    assert_eq!(outcome.miss_count, 1);
}

#[test]
fn should_never_give_up_an_expert_of_the_route_being_served() {
    // One token routes to two experts with room for one: the second cannot
    // displace the first, which the same multiply still needs.
    let token_routes = vec![vec![vec![0, 1]], vec![vec![0, 1]]];
    let replay_case = ReplayCase {
        routed_reply: RoutedReply::from_token_routes(8, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES],
        capacity_bytes: EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![Vec::new()],
        layer_slot_limits: None,
        pinned_layer_indexes: Vec::new(),
    };

    let outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert_eq!(outcome.miss_count, 3);
    assert_eq!(outcome.hit_count, 1);
}

#[test]
fn should_refuse_a_newcomer_used_less_often_than_its_victim() {
    // Room for one. Expert 0 is used three times before 1 appears once, so
    // frequency admission keeps 0 and the final use of 0 is a hit.
    let replay_case = single_layer_case(&[0, 0, 0, 1, 0], 1);

    let admitted_outcome = replay(&replay_case, &mut FrequencyAdmittedRecency::default());
    let recency_outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert_eq!(admitted_outcome.miss_count, 2);
    assert_eq!(recency_outcome.miss_count, 3);
}

#[test]
fn should_keep_a_layer_within_its_slot_limit_even_with_memory_to_spare() {
    // Two layers, room for four experts in all, but layer 0 may hold only
    // one: its second expert keeps displacing its first.
    let token_routes = vec![
        vec![vec![0], vec![5]],
        vec![vec![1], vec![5]],
        vec![vec![0], vec![5]],
        vec![vec![1], vec![5]],
    ];
    let shared_case = ReplayCase {
        routed_reply: RoutedReply::from_token_routes(8, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES, EXPERT_BYTES],
        capacity_bytes: 4 * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![Vec::new(), Vec::new()],
        layer_slot_limits: None,
        pinned_layer_indexes: Vec::new(),
    };
    let divided_case = ReplayCase {
        layer_slot_limits: Some(vec![1, 4]),
        ..shared_case.clone()
    };

    let shared_outcome = replay(&shared_case, &mut LeastRecentlyUsed::default());
    let divided_outcome = replay(&divided_case, &mut LeastRecentlyUsed::default());

    assert_eq!(shared_outcome.miss_count, 3);
    assert_eq!(divided_outcome.miss_count, 5);
}

#[test]
fn should_never_keep_an_expert_of_a_layer_whose_limit_is_zero() {
    let mut replay_case = single_layer_case(&[0, 0, 0], 3);
    replay_case.layer_slot_limits = Some(vec![0]);

    assert_eq!(
        replay(&replay_case, &mut LeastRecentlyUsed::default()).miss_count,
        3
    );
}

#[test]
fn should_never_give_up_an_expert_of_a_pinned_layer() {
    // Layer 0 starts holding expert 0 and is pinned; layer 1 keeps asking for
    // new experts, which may take only room the pinned layer does not hold.
    let token_routes = vec![
        vec![vec![1], vec![2]],
        vec![vec![1], vec![3]],
        vec![vec![0], vec![2]],
    ];
    let replay_case = ReplayCase {
        routed_reply: RoutedReply::from_token_routes(8, &token_routes).expect("routes build"),
        expert_payload_bytes_by_layer: vec![EXPERT_BYTES, EXPERT_BYTES],
        capacity_bytes: 2 * EXPERT_BYTES,
        held_expert_ids_by_layer_at_start: vec![vec![0], Vec::new()],
        layer_slot_limits: None,
        pinned_layer_indexes: vec![0],
    };

    let outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    // Layer 0 also keeps expert 1 in its free room (a hit at the second
    // token); layer 1 may never take a pinned expert's room, so its three
    // uses all miss, and expert 0 survives to its use at the last token.
    assert_eq!(outcome.hit_count, 2);
    assert_eq!(outcome.miss_count, 4);
}

#[test]
fn should_seat_no_more_starting_experts_in_a_layer_than_its_slot_limit() {
    let mut replay_case = single_layer_case(&[0, 1, 2], 3);
    replay_case.held_expert_ids_by_layer_at_start = vec![vec![0, 1, 2]];
    replay_case.layer_slot_limits = Some(vec![1]);

    let outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    // Only expert 0 is seated; 1 and 2 are read.
    assert_eq!(outcome.hit_count, 1);
    assert_eq!(outcome.miss_count, 2);
}
