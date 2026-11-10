//! Unit tests for grouping question sequences into bounded forward passes.

use super::{PADDED_TOKENS_PER_PASS, plan_forward_passes};

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
    assert!(plan_forward_passes(&[], PADDED_TOKENS_PER_PASS).is_empty());
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
    let forward_passes = plan_forward_passes(&sequence_lengths, PADDED_TOKENS_PER_PASS);
    let expected_indices: Vec<usize> = (0..257).collect();
    assert_eq!(covered_indices(&forward_passes), expected_indices);
    for forward_pass in &forward_passes {
        let mut longest_length = 0;
        for sequence_index in forward_pass {
            longest_length = longest_length.max(sequence_lengths[*sequence_index]);
        }
        assert!(longest_length * forward_pass.len() <= PADDED_TOKENS_PER_PASS);
    }
}
