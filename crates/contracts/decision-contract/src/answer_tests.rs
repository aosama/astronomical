//! Unit tests for typed answers and the two confidence numbers.

use super::*;
use crate::test_support::{assert_close, single_question};

#[test]
fn should_choose_the_most_probable_option_and_report_every_probability() {
    let question = single_question(
        r#"{"team": {"type": "choice", "instructions": "Who?", "criteria": ["a", "b", "c"]}}"#,
    );
    let answer = answer_from_probabilities(&question, &[0.2, 0.7, 0.1]).expect("an answer");
    let DecisionAnswer::Choice(choice_answer) = answer else {
        panic!("a choice answer");
    };
    assert_eq!(choice_answer.chosen_key, "b");
    assert_eq!(choice_answer.option_probabilities[2].key, "c");
    assert_close(choice_answer.option_probabilities[2].probability, 0.1, 0.0);
    assert_close(choice_answer.answer_confidence, 0.7, 0.0);
}

#[test]
fn should_choose_the_first_option_when_probabilities_tie() {
    let question = single_question(
        r#"{"team": {"type": "choice", "instructions": "Who?", "criteria": ["a", "b"]}}"#,
    );
    let DecisionAnswer::Choice(choice_answer) =
        answer_from_probabilities(&question, &[0.5, 0.5]).expect("an answer")
    else {
        panic!("a choice answer");
    };
    assert_eq!(choice_answer.chosen_key, "a");
}

#[test]
fn should_report_a_score_between_levels_with_its_legend() {
    let question = single_question(
        r#"{"urgency": {"type": "score", "criteria": ["low", {"label": "mid"}, "high"]}}"#,
    );
    let DecisionAnswer::Score(score_answer) =
        answer_from_probabilities(&question, &[0.25, 0.5, 0.25]).expect("an answer")
    else {
        panic!("a score answer");
    };
    assert_close(score_answer.expected_level, 1.0, 1e-12);
    assert_eq!(
        score_answer.level_legend,
        ["low", r#"{"label": "mid"}"#, "high"]
    );
    assert_eq!(score_answer.level_probabilities, [0.25, 0.5, 0.25]);
}

#[test]
fn should_report_the_probability_of_true_from_the_canonical_order() {
    let question = single_question(r#"{"refund": {"type": "noul", "instructions": "Asked?"}}"#);
    let DecisionAnswer::Noul(noul_answer) =
        answer_from_probabilities(&question, &[0.8, 0.2]).expect("an answer")
    else {
        panic!("a noul answer");
    };
    assert_close(noul_answer.probability_true, 0.2, 0.0);
    assert_close(noul_answer.answer_confidence, 0.8, 0.0);
}

#[test]
fn should_reject_a_probability_count_that_differs_from_the_option_count() {
    let question = single_question(
        r#"{"team": {"type": "choice", "instructions": "Who?", "criteria": ["a", "b"]}}"#,
    );
    assert_eq!(
        answer_from_probabilities(&question, &[1.0]),
        Err(AnswerError::ProbabilityCountMismatch {
            question: "team".to_owned(),
            expected_option_count: 2,
            actual_probability_count: 1,
        })
    );
}

#[test]
fn should_reject_a_question_built_with_no_options() {
    let question = DecisionQuestion {
        name: "empty".to_owned(),
        instructions: None,
        body: DecisionQuestionBody::Choice {
            options: Vec::new(),
        },
    };
    assert_eq!(
        answer_from_probabilities(&question, &[]),
        Err(AnswerError::NoOptions {
            question: "empty".to_owned(),
        })
    );
}

#[test]
fn should_report_full_distribution_confidence_for_one_certain_option() {
    assert_close(distribution_confidence(&[1.0, 0.0, 0.0]), 1.0, 1e-9);
}

#[test]
fn should_report_no_distribution_confidence_for_an_even_spread() {
    assert_close(
        distribution_confidence(&[0.25, 0.25, 0.25, 0.25]),
        0.0,
        1e-12,
    );
}

#[test]
fn should_report_full_confidence_for_fewer_than_two_options() {
    assert_close(distribution_confidence(&[1.0]), 1.0, 0.0);
    assert_close(distribution_confidence(&[]), 1.0, 0.0);
    assert_close(answer_confidence(&[]), 1.0, 0.0);
}

#[test]
fn should_report_the_largest_probability_as_answer_confidence() {
    assert_close(answer_confidence(&[0.1, 0.6, 0.3]), 0.6, 0.0);
}
