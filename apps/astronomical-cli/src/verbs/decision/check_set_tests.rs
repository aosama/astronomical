//! Unit tests for the built-in check set.

use decision_contract::{DecisionQuestionBody, parse_system_one_questions};

use super::*;

#[test]
fn should_hold_one_hundred_and_twenty_valid_questions_of_every_kind_and_width() {
    let check_examples = built_in_check_examples().unwrap();
    assert_eq!(check_examples.len(), 24);
    let mut option_counts = Vec::new();
    let mut question_total = 0;
    for check_example in &check_examples {
        let questions = parse_system_one_questions(&check_example.questions_document).unwrap();
        question_total += questions.len();
        for question in &questions {
            option_counts.push(question.option_count());
        }
    }
    assert_eq!(question_total, 120);
    option_counts.sort_unstable();
    option_counts.dedup();
    assert_eq!(option_counts, vec![2, 3, 4, 7, 12]);
    let first_questions =
        parse_system_one_questions(&check_examples[0].questions_document).unwrap();
    assert!(matches!(
        first_questions[0].body,
        DecisionQuestionBody::Noul { .. }
    ));
}

#[test]
fn should_ask_about_text_states_only() {
    for check_example in built_in_check_examples().unwrap() {
        assert!(check_example.state.as_text().is_some());
    }
}
