//! The shuffle against values printed by numpy 2.5.4:
//! `r = np.random.RandomState(seed); s = list(range(n)); r.shuffle(s)`.

use super::*;

fn shuffled_range(length: usize, seed: u64) -> Vec<usize> {
    let mut items: Vec<usize> = (0..length).collect();
    shuffle_like_numpy_random_state(&mut items, seed);
    items
}

#[test]
fn should_reproduce_numpy_for_ten_items_at_seed_forty_two() {
    assert_eq!(shuffled_range(10, 42), vec![8, 1, 5, 0, 7, 2, 9, 4, 3, 6]);
}

#[test]
fn should_reproduce_numpy_for_thirty_seven_items_at_seed_forty_two() {
    assert_eq!(
        shuffled_range(37, 42),
        vec![
            17, 13, 4, 29, 35, 25, 6, 26, 24, 16, 8, 15, 12, 32, 9, 19, 0, 31, 5, 11, 1, 27, 21, 2,
            36, 34, 3, 33, 23, 30, 10, 22, 18, 20, 7, 14, 28
        ]
    );
}

#[test]
fn should_reproduce_numpy_for_another_seed() {
    assert_eq!(
        shuffled_range(20, 7),
        vec![
            1, 17, 2, 5, 11, 0, 18, 6, 13, 19, 10, 14, 8, 16, 9, 12, 7, 3, 4, 15
        ]
    );
}

#[test]
fn should_leave_zero_and_one_item_slices_unchanged() {
    assert_eq!(shuffled_range(1, 42), vec![0]);
    assert!(shuffled_range(0, 42).is_empty());
}

#[test]
fn should_draw_only_zero_from_a_zero_wide_interval() {
    let mut generator = MersenneTwister19937::from_seed(42);

    assert_eq!(generator.next_in_interval(0), 0);
}
