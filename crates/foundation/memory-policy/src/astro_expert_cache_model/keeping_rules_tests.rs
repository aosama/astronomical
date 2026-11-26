//! Keeping rules on their own, against traces whose answers are known: the
//! engine's baseline rule (`FewestReadsSinceSeated`).

use super::replay_lab_tests::single_layer_case;
use super::{FewestReadsSinceSeated, KeepingRule, LeastRecentlyUsed, replay};

#[test]
fn should_let_each_newcomer_replace_the_last_rather_than_an_expert_read_again() {
    // Room for two. Expert 0 is read again after it is seated; 1, 2 and 3
    // are each read once, so each replaces the one before it and 0 stays.
    let replay_case = single_layer_case(&[0, 0, 1, 2, 3, 0], 2);

    let engine_outcome = replay(&replay_case, &mut FewestReadsSinceSeated::default());
    let recency_outcome = replay(&replay_case, &mut LeastRecentlyUsed::default());

    assert_eq!(engine_outcome.hit_count, 2);
    assert_eq!(engine_outcome.miss_count, 4);
    assert_eq!(recency_outcome.hit_count, 1);
}

#[test]
fn should_replace_a_held_expert_however_often_it_was_read_when_it_is_the_only_one() {
    // Room for one. Expert 0 is read three times, then 1 arrives: the only
    // victim is 0, whatever its reads, because the engine seats every
    // newcomer it has a slot or a victim for.
    let replay_case = single_layer_case(&[0, 0, 0, 1, 1, 1], 1);

    let outcome = replay(&replay_case, &mut FewestReadsSinceSeated::default());

    // 0 is read, then hit twice; 1 replaces it, then is hit twice.
    assert_eq!(outcome.miss_count, 2);
    assert_eq!(outcome.hit_count, 4);
}

#[test]
fn should_protect_experts_the_prompt_read_most_when_the_reply_begins() {
    // Room for two, both held at the start. The prompt read 0 often and 1
    // once, so the newcomer 2 replaces 1 and 0 survives to its use.
    let mut replay_case = single_layer_case(&[2, 0], 2);
    replay_case.held_expert_ids_by_layer_at_start = vec![vec![0, 1]];
    let mut seeded_rule = FewestReadsSinceSeated::seeded_with_prompt_demand(&[vec![5, 1]]);

    let outcome = replay(&replay_case, &mut seeded_rule);

    assert_eq!(outcome.hit_count, 1);
    assert_eq!(outcome.miss_count, 1);
    assert!(seeded_rule.name().contains("prompt"));
}

#[test]
fn should_count_nothing_for_an_empty_reply() {
    let replay_case = single_layer_case(&[], 2);

    let outcome = replay(&replay_case, &mut FewestReadsSinceSeated::default());

    assert_eq!(outcome.hit_count + outcome.miss_count, 0);
}

#[test]
fn should_miss_every_use_when_nothing_can_be_kept() {
    let replay_case = single_layer_case(&[0, 0, 0], 0);

    let outcome = replay(&replay_case, &mut FewestReadsSinceSeated::default());

    assert_eq!(outcome.miss_count, 3);
}
