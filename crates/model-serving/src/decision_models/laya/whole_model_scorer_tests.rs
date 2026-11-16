//! How large a whole-model training's micro-batches are on Macs of
//! different sizes.

use super::*;

const GIGABYTE: usize = 1 << 30;

#[test]
fn should_use_the_largest_micro_batch_on_a_mac_with_room() {
    // A 48 GB Mac's runtime limit is about 35 GB.
    assert_eq!(training_tokens_per_pass(35 * GIGABYTE), 1_024);
}

#[test]
fn should_shrink_micro_batches_on_a_mac_with_little_memory() {
    // An 8 GB Mac's runtime limit is about 5 GB: a quarter of it holds
    // about 350 tokens' activations.
    let tokens_per_pass = training_tokens_per_pass(5 * GIGABYTE);

    assert!(tokens_per_pass < 400, "{tokens_per_pass}");
    assert!(tokens_per_pass * TRAINING_BYTES_PER_PADDED_TOKEN <= 5 * GIGABYTE / 4);
}

#[test]
fn should_never_go_below_one_long_question_however_small_the_limit() {
    assert_eq!(training_tokens_per_pass(0), 128);
    assert_eq!(training_tokens_per_pass(GIGABYTE / 8), 128);
}
