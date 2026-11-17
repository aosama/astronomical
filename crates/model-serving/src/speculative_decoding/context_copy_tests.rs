use super::context_copy::{ContextCopier, ContextCopySettings};

fn copier(minimum_match: usize) -> ContextCopier {
    ContextCopier::new(ContextCopySettings {
        ngram: 2,
        minimum_match,
        ..ContextCopySettings::default()
    })
}

#[test]
fn should_copy_what_followed_an_earlier_occurrence_of_the_tail() {
    // TensorFold's own example: the tail 1 2 3 appeared before, then 4 5 9.
    let history = [1, 2, 3, 4, 5, 9, 1, 2, 3];
    assert_eq!(copier(2).propose(&history, 3), vec![4, 5, 9]);
}

#[test]
fn should_stop_at_the_requested_count_and_at_the_end_of_the_history() {
    let history = [7, 8, 1, 2, 7, 8];
    assert_eq!(copier(2).propose(&history, 1), vec![1]);
    assert_eq!(copier(2).propose(&history, 10), vec![1, 2, 7, 8]);
}

#[test]
fn should_prefer_the_occurrence_whose_match_runs_furthest_back() {
    // 5 6 occurs twice; only the first is preceded by 4, as the tail is.
    let history = [4, 5, 6, 10, 9, 5, 6, 20, 4, 5, 6];
    assert_eq!(copier(2).propose(&history, 1), vec![10]);
}

#[test]
fn should_propose_nothing_when_the_best_match_is_too_short() {
    let history = [9, 1, 2, 30, 8, 1, 2];
    assert_eq!(copier(3).propose(&history, 4), Vec::<u32>::new());
    assert_eq!(copier(2).propose(&history, 1), vec![30]);
}

#[test]
fn should_propose_nothing_for_a_short_history_or_no_room() {
    assert!(copier(2).propose(&[1, 2], 4).is_empty());
    assert!(copier(2).propose(&[1, 2, 3, 1, 2], 0).is_empty());
}

#[test]
fn should_go_silent_after_a_window_of_proposals_that_keep_nothing() {
    let history = [1, 2, 3, 4, 1, 2];
    let mut copier = ContextCopier::new(ContextCopySettings {
        ngram: 2,
        minimum_match: 2,
        window: 2,
        silence_asks: 3,
        ..ContextCopySettings::default()
    });
    copier.observe(4, 0);
    copier.observe(4, 0);
    for _ in 0..3 {
        assert!(copier.propose(&history, 2).is_empty());
    }
    assert_eq!(copier.propose(&history, 2), vec![3, 4]);
}

#[test]
fn should_stay_active_while_some_copies_land() {
    let history = [1, 2, 3, 4, 1, 2];
    let mut copier = ContextCopier::new(ContextCopySettings {
        ngram: 2,
        minimum_match: 2,
        window: 2,
        ..ContextCopySettings::default()
    });
    copier.observe(4, 0);
    copier.observe(4, 1);
    assert_eq!(copier.propose(&history, 2), vec![3, 4]);
}
