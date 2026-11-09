//! Kept-layer counts after releases and re-plans: nothing to release, an
//! exact layer, a partial layer, more than is kept, and both re-plan
//! directions.

use super::{
    KeptLayerReplan, kept_layer_count_after_release, kept_layer_count_after_replan,
    streaming_slot_count_in_use,
};

const LAYER_BYTES: [u64; 4] = [100, 100, 150, 100];

#[test]
fn should_release_nothing_when_no_bytes_are_required() {
    assert_eq!(kept_layer_count_after_release(&LAYER_BYTES, 4, 0), (4, 0));
}

#[test]
fn should_release_exactly_one_layer_for_exactly_its_bytes() {
    assert_eq!(
        kept_layer_count_after_release(&LAYER_BYTES, 4, 100),
        (3, 100)
    );
}

#[test]
fn should_release_a_whole_layer_for_one_byte() {
    assert_eq!(kept_layer_count_after_release(&LAYER_BYTES, 4, 1), (3, 100));
}

#[test]
fn should_release_trailing_layers_until_the_bytes_are_covered() {
    assert_eq!(
        kept_layer_count_after_release(&LAYER_BYTES, 4, 101),
        (2, 250)
    );
}

#[test]
fn should_stop_at_zero_kept_layers_when_more_is_required_than_kept() {
    assert_eq!(
        kept_layer_count_after_release(&LAYER_BYTES, 2, 10_000),
        (0, 200)
    );
}

#[test]
fn should_release_nothing_when_no_layer_is_kept() {
    assert_eq!(kept_layer_count_after_release(&LAYER_BYTES, 0, 500), (0, 0));
}

#[test]
fn should_follow_the_plan_up_and_down_when_matching_it() {
    assert_eq!(
        kept_layer_count_after_replan(7, 3, KeptLayerReplan::MatchPlan),
        7
    );
    assert_eq!(
        kept_layer_count_after_replan(3, 7, KeptLayerReplan::MatchPlan),
        3
    );
}

#[test]
fn should_never_read_layers_back_before_a_lower_limit() {
    assert_eq!(
        kept_layer_count_after_replan(7, 3, KeptLayerReplan::NeverGrow),
        3
    );
}

#[test]
fn should_give_layers_up_before_a_lower_limit() {
    assert_eq!(
        kept_layer_count_after_replan(3, 7, KeptLayerReplan::NeverGrow),
        3
    );
    assert_eq!(
        kept_layer_count_after_replan(5, 5, KeptLayerReplan::NeverGrow),
        5
    );
}

#[test]
fn should_hold_no_slots_while_every_layer_is_kept() {
    assert_eq!(streaming_slot_count_in_use(32, 32, 2), 0);
}

#[test]
fn should_hold_one_slot_when_one_layer_streams() {
    assert_eq!(streaming_slot_count_in_use(32, 31, 2), 1);
}

#[test]
fn should_hold_every_slot_when_many_layers_stream() {
    assert_eq!(streaming_slot_count_in_use(32, 0, 2), 2);
}
