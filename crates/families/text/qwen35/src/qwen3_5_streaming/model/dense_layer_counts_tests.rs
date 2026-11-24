//! Kept-layer counts after releases and for a given room: nothing to
//! release, an exact layer, a partial layer, more than is kept, and the
//! leading layers that fit.

use super::{
    kept_layer_count_after_release, leading_layer_count_within, streaming_slot_count_in_use,
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
fn should_keep_every_leading_layer_that_fits_the_room() {
    assert_eq!(leading_layer_count_within(&LAYER_BYTES, 350), 3);
    assert_eq!(leading_layer_count_within(&LAYER_BYTES, 10_000), 4);
}

#[test]
fn should_stop_at_the_first_layer_that_does_not_fit() {
    // The third layer needs 150; a later 100-byte layer is not kept instead.
    assert_eq!(leading_layer_count_within(&LAYER_BYTES, 349), 2);
    assert_eq!(leading_layer_count_within(&LAYER_BYTES, 99), 0);
}

#[test]
fn should_keep_no_layer_in_no_room_or_of_no_layers() {
    assert_eq!(leading_layer_count_within(&LAYER_BYTES, 0), 0);
    assert_eq!(leading_layer_count_within(&[], 1_000), 0);
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
