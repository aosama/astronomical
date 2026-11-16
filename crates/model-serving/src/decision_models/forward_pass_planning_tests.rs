//! Unit tests for grouping question sequences into bounded forward passes.

use super::{LAYA_SERVING_PADDED_TOKENS_PER_PASS, plan_forward_passes, training_tokens_per_pass};

fn covered_indices(forward_passes: &[Vec<usize>]) -> Vec<usize> {
    let mut covered_indices = Vec::new();
    for forward_pass in forward_passes {
        covered_indices.extend_from_slice(forward_pass);
    }
    covered_indices.sort_unstable();
    covered_indices
}

#[test]
fn should_plan_no_passes_for_no_sequences() {
    assert!(plan_forward_passes(&[], LAYA_SERVING_PADDED_TOKENS_PER_PASS).is_empty());
}

#[test]
fn should_put_everything_in_one_pass_when_it_fits() {
    let forward_passes = plan_forward_passes(&[10, 30, 20], 100);
    assert_eq!(forward_passes.len(), 1);
    assert_eq!(covered_indices(&forward_passes), [0, 1, 2]);
}

#[test]
fn should_fill_a_pass_to_exactly_its_budget_and_start_another_one_token_over() {
    // Four sequences of 25 tokens are exactly 100 padded tokens.
    assert_eq!(plan_forward_passes(&[25, 25, 25, 25], 100).len(), 1);
    // Padding all four to 26 tokens would need 104.
    let forward_passes = plan_forward_passes(&[25, 25, 25, 26], 100);
    assert_eq!(forward_passes.len(), 2);
    assert_eq!(forward_passes[0], [0, 1, 2]);
    assert_eq!(forward_passes[1], [3]);
}

#[test]
fn should_group_sequences_of_similar_length_together() {
    let forward_passes = plan_forward_passes(&[90, 10, 95, 12], 200);
    assert_eq!(forward_passes, [vec![1, 3], vec![0, 2]]);
}

#[test]
fn should_give_a_sequence_longer_than_the_budget_a_pass_of_its_own() {
    let forward_passes = plan_forward_passes(&[500, 10], 100);
    assert_eq!(forward_passes, [vec![1], vec![0]]);
}

#[test]
fn should_place_every_sequence_in_exactly_one_pass() {
    let mut sequence_lengths = Vec::new();
    for sequence_index in 0..257 {
        sequence_lengths.push(1 + (sequence_index * 37) % 512);
    }
    let forward_passes =
        plan_forward_passes(&sequence_lengths, LAYA_SERVING_PADDED_TOKENS_PER_PASS);
    let expected_indices: Vec<usize> = (0..257).collect();
    assert_eq!(covered_indices(&forward_passes), expected_indices);
    for forward_pass in &forward_passes {
        let mut longest_length = 0;
        for sequence_index in forward_pass {
            longest_length = longest_length.max(sequence_lengths[*sequence_index]);
        }
        assert!(longest_length * forward_pass.len() <= LAYA_SERVING_PADDED_TOKENS_PER_PASS);
    }
}

const GIGABYTE: usize = 1 << 30;
/// Laya's measured activation cost per padded token.
const LAYA_BYTES_PER_TOKEN: usize = 3_800_000;

#[test]
fn should_use_the_largest_training_micro_batch_on_a_mac_with_room() {
    // A 48 GB Mac's runtime limit is about 35 GB.
    assert_eq!(
        training_tokens_per_pass(35 * GIGABYTE, LAYA_BYTES_PER_TOKEN),
        1_024
    );
}

#[test]
fn should_shrink_training_micro_batches_on_a_mac_with_little_memory() {
    // An 8 GB Mac's runtime limit is about 5 GB: a quarter of it holds
    // about 350 of Laya's tokens.
    let tokens_per_pass = training_tokens_per_pass(5 * GIGABYTE, LAYA_BYTES_PER_TOKEN);

    assert!(tokens_per_pass < 400, "{tokens_per_pass}");
    assert!(tokens_per_pass * LAYA_BYTES_PER_TOKEN <= 5 * GIGABYTE / 4);
}

#[test]
fn should_give_a_larger_model_smaller_training_micro_batches() {
    // Clef's 9B keeps about 16 MB per token: a 35 GB limit holds 546.
    let clef_tokens = training_tokens_per_pass(35 * GIGABYTE, 17_000_000);

    assert!((512..1_024).contains(&clef_tokens), "{clef_tokens}");
}

#[test]
fn should_never_go_below_one_long_question_however_small_the_limit() {
    assert_eq!(training_tokens_per_pass(0, LAYA_BYTES_PER_TOKEN), 128);
    assert_eq!(
        training_tokens_per_pass(GIGABYTE / 8, LAYA_BYTES_PER_TOKEN),
        128
    );
    assert_eq!(training_tokens_per_pass(GIGABYTE, 0), 1_024);
}
