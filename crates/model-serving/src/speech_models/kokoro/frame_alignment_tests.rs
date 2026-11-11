use super::phoneme_index_per_frame;

#[test]
fn should_repeat_each_phoneme_for_its_frames() {
    assert_eq!(phoneme_index_per_frame(&[2, 1, 3]), [0, 0, 1, 2, 2, 2]);
}

// Boundary: a single phoneme of one frame.
#[test]
fn should_align_a_single_frame() {
    assert_eq!(phoneme_index_per_frame(&[1]), [0]);
}

// Negative: a non-positive count contributes no frames.
#[test]
fn should_skip_non_positive_counts() {
    assert_eq!(phoneme_index_per_frame(&[0, -2, 1]), [2]);
}
