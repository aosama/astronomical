//! Unit tests for the split by state.

use decision_contract::{DecisionQuestionKind, OrderedJsonValue};

use super::*;
use crate::test_support::{ROMEO_AND_JULIET_LINES, scored_question};

fn questions_over_states(state_count: u64, questions_per_state: usize) -> Vec<ScoredQuestion> {
    let mut scored_questions = Vec::new();
    for state_key in 0..state_count {
        for _ in 0..questions_per_state {
            scored_questions.push(scored_question(
                DecisionQuestionKind::Noul,
                &[0.0, 1.0],
                1,
                state_key,
            ));
        }
    }
    scored_questions
}

#[test]
fn should_give_equal_states_equal_keys_and_different_states_different_keys() {
    let romeo = OrderedJsonValue::Text(ROMEO_AND_JULIET_LINES[2].to_owned());
    let same_romeo = OrderedJsonValue::Text(ROMEO_AND_JULIET_LINES[2].to_owned());
    let juliet = OrderedJsonValue::Text(ROMEO_AND_JULIET_LINES[1].to_owned());
    assert_eq!(state_group_key(&romeo), state_group_key(&same_romeo));
    assert_ne!(state_group_key(&romeo), state_group_key(&juliet));
}

#[test]
fn should_keep_the_key_stable_across_runs() {
    // Pinned so a change of hash function, which would silently move states
    // between sides, fails here first.
    // The expected value is FNV-1a 64 of the JSON text, quotes included,
    // computed independently of this crate.
    let romeo = OrderedJsonValue::Text(ROMEO_AND_JULIET_LINES[2].to_owned());
    assert_eq!(state_group_key(&romeo), 0x90cf_0c2b_5872_942d);
}

#[test]
fn should_never_put_one_state_on_both_sides() {
    let holdout_split = split_by_state(questions_over_states(50, 3));
    for holdout_question in &holdout_split.holdout_questions {
        for fitting_question in &holdout_split.fitting_questions {
            assert_ne!(
                holdout_question.state_group_key,
                fitting_question.state_group_key
            );
        }
    }
}

#[test]
fn should_hold_out_one_state_in_five() {
    let holdout_split = split_by_state(questions_over_states(50, 3));
    assert_eq!(holdout_split.holdout_questions.len(), 10 * 3);
    assert_eq!(holdout_split.fitting_questions.len(), 40 * 3);
}

#[test]
fn should_split_the_same_questions_the_same_way_every_time() {
    let first_split = split_by_state(questions_over_states(23, 2));
    let second_split = split_by_state(questions_over_states(23, 2));
    assert_eq!(first_split, second_split);
}

#[test]
fn should_hold_out_nothing_from_fewer_than_five_states() {
    let holdout_split = split_by_state(questions_over_states(4, 10));
    assert!(holdout_split.holdout_questions.is_empty());
    assert_eq!(holdout_split.fitting_questions.len(), 40);
}

#[test]
fn should_split_no_questions_into_two_empty_sides() {
    assert_eq!(split_by_state(Vec::new()), HoldoutSplit::default());
}
