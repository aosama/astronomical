//! Unit tests for the question model.

use super::*;
use crate::test_support::single_question;

#[test]
fn should_round_trip_every_kind_through_its_wire_name() {
    for kind in [
        DecisionQuestionKind::Choice,
        DecisionQuestionKind::Score,
        DecisionQuestionKind::Noul,
    ] {
        assert_eq!(
            DecisionQuestionKind::from_wire_name(kind.wire_name()),
            Some(kind)
        );
    }
}

#[test]
fn should_not_recognise_an_unknown_or_differently_cased_wire_name() {
    assert_eq!(DecisionQuestionKind::from_wire_name("ranking"), None);
    assert_eq!(DecisionQuestionKind::from_wire_name("Choice"), None);
    assert_eq!(DecisionQuestionKind::from_wire_name(""), None);
}

#[test]
fn should_count_the_answers_each_kind_allows() {
    let choice = single_question(r#"{"q": {"type": "choice", "criteria": ["a", "b", "c"]}}"#);
    let score = single_question(r#"{"q": {"type": "score", "criteria": ["low"]}}"#);
    let noul = single_question(r#"{"q": {"type": "noul"}}"#);
    assert_eq!(choice.option_count(), 3);
    assert_eq!(
        score.option_count(),
        1,
        "a single level is the smallest scale"
    );
    assert_eq!(noul.option_count(), NOUL_OPTION_COUNT);
}

#[test]
fn should_order_noul_probabilities_as_false_then_true() {
    assert_eq!(NOUL_FALSE_INDEX, 0);
    assert_eq!(NOUL_TRUE_INDEX, 1);
}
