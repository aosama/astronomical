//! That a recomputed layer trains to the same weights as a stored one needs
//! a real Qwen3.5 checkpoint (the crate has no synthetic one): it is proven
//! by `cargo test-kev-whole-model-layer-recomputation` and its Clef alias.
//! These cover what needs no model.

use super::*;

#[test]
fn should_accept_a_down_and_an_up_array_for_every_target() {
    assert!(check_adapter_input_count(14, 7).is_ok());
    assert!(check_adapter_input_count(0, 0).is_ok());
}

#[test]
fn should_refuse_an_add_on_input_count_that_does_not_pair_with_the_targets() {
    let failure = check_adapter_input_count(13, 7).unwrap_err();

    assert_eq!(failure.operation, "recompute a decoder layer");
    assert!(
        failure
            .description
            .contains("13 add-on arrays for 7 targets")
    );
    assert!(check_adapter_input_count(16, 7).is_err());
}

#[test]
fn should_count_each_recomputed_text_once() {
    let recomputed_texts = RecomputedTexts::default();

    recomputed_texts.record(3);
    recomputed_texts.record(3);
    recomputed_texts.record(8);

    assert_eq!(recomputed_texts.count(), 2);
}

#[test]
fn should_count_no_text_before_any_was_recomputed() {
    assert_eq!(RecomputedTexts::default().count(), 0);
}
