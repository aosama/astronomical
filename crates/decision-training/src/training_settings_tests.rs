//! Unit tests for the shared pieces of training.

use super::*;

#[test]
fn should_shuffle_into_a_reproducible_permutation_that_differs_by_seed() {
    let first = shuffled_order(50, 1, 3);
    let mut sorted = first.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<usize>>());
    assert_eq!(first, shuffled_order(50, 1, 3));
    assert_ne!(first, shuffled_order(50, 2, 3));
    assert_ne!(first, shuffled_order(50, 1, 4));
    assert!(shuffled_order(0, 1, 1).is_empty());
}
