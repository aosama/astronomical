//! The pure parts of the top-p mask: reading the candidate grid and deciding
//! when the mask filters nothing. The GPU graph itself is exercised by the
//! sampler tests that run it on a real runtime.

use super::{CandidateGrid, top_p_filters_nothing};

#[test]
fn should_read_the_row_count_and_top_k_from_a_three_axis_candidate_shape() {
    let candidate_grid = CandidateGrid::from_shape(&[1, 16, 20]).expect("a [1, 16, 20] grid");

    assert_eq!(
        candidate_grid,
        CandidateGrid {
            row_count: 16,
            top_k: 20,
        }
    );
    assert_eq!(candidate_grid.shape(), [1, 16, 20]);
    assert_eq!(candidate_grid.element_count(), 320);
}

#[test]
fn should_accept_a_single_row_which_is_the_smallest_positive_row_count() {
    let candidate_grid = CandidateGrid::from_shape(&[1, 1, 32]).expect("a single-row grid");

    assert_eq!(candidate_grid.row_count, 1);
    assert_eq!(candidate_grid.element_count(), 32);
}

#[test]
fn should_refuse_a_candidate_shape_that_is_not_one_by_rows_by_top_k() {
    for refused_shape in [
        &[1, 20][..],
        &[2, 4, 20][..],
        &[1, 0, 20][..],
        &[1, 4, 20, 1][..],
        &[][..],
    ] {
        let refusal = CandidateGrid::from_shape(refused_shape)
            .expect_err("a malformed candidate shape should be refused");

        assert_eq!(
            refusal.reason,
            "selected probabilities must have shape [1, rows, top_k] with a positive row count",
            "{refused_shape:?}"
        );
    }
}

#[test]
fn should_filter_nothing_when_top_p_is_unset_or_keeps_the_whole_distribution() {
    assert!(top_p_filters_nothing(0));
    assert!(top_p_filters_nothing(1_000));
    assert!(top_p_filters_nothing(1_500));
}

#[test]
fn should_filter_candidates_for_any_top_p_strictly_between_zero_and_one() {
    assert!(!top_p_filters_nothing(1));
    assert!(!top_p_filters_nothing(950));
    assert!(!top_p_filters_nothing(999));
}
