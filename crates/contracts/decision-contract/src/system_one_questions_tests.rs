//! Unit tests for System One question parsing and validation.

use super::*;
use crate::test_support::{ordered_json, single_question};

fn parse_error(questions_json_text: &str) -> QuestionDefinitionError {
    parse_system_one_questions(&ordered_json(questions_json_text))
        .expect_err("the questions are invalid")
}

fn repeated_labels_json(label_count: usize) -> String {
    let mut quoted_labels = Vec::with_capacity(label_count);
    for label_index in 0..label_count {
        quoted_labels.push(format!("\"label{label_index}\""));
    }
    format!(
        r#"{{"q": {{"type": "choice", "instructions": "Pick.", "criteria": [{}]}}}}"#,
        quoted_labels.join(",")
    )
}

#[test]
fn should_parse_a_choice_with_described_options_in_request_order() {
    let question = single_question(
        r#"{"team": {"type": "choice", "instructions": "Who owns it?",
            "criteria": {"support": "bugs", "billing": "refunds", "other": ""}}}"#,
    );
    assert_eq!(question.name, "team");
    assert_eq!(question.kind(), DecisionQuestionKind::Choice);
    let DecisionQuestionBody::Choice { options } = &question.body else {
        panic!("a choice body");
    };
    assert_eq!(options[0].key, "support");
    assert_eq!(options[1].key, "billing");
    assert_eq!(
        options[1].description,
        Some(OrderedJsonValue::Text("refunds".to_owned()))
    );
    assert_eq!(
        options[2].description, None,
        "an empty description is no description"
    );
}

#[test]
fn should_parse_a_choice_given_as_a_list_of_labels() {
    let question = single_question(
        r#"{"mood": {"type": "choice", "instructions": "Pick.", "criteria": ["grave", "merry"]}}"#,
    );
    assert_eq!(question.option_count(), 2);
}

#[test]
fn should_parse_a_score_and_a_noul_without_instructions() {
    let parsed_questions = parse_system_one_questions(&ordered_json(
        r#"{"urgency": {"type": "score", "criteria": ["low", "high"]},
            "refund": {"type": "noul"}}"#,
    ))
    .expect("valid questions");
    assert_eq!(parsed_questions[0].kind(), DecisionQuestionKind::Score);
    assert_eq!(parsed_questions[0].instructions, None);
    assert_eq!(parsed_questions[1].kind(), DecisionQuestionKind::Noul);
    assert_eq!(parsed_questions[1].option_count(), 2);
}

#[test]
fn should_read_noul_descriptions_whatever_the_case_of_their_keys() {
    let question = single_question(
        r#"{"q": {"type": "noul", "instructions": "Holds?",
            "criteria": {"True": "it holds", "FALSE": null}}}"#,
    );
    let DecisionQuestionBody::Noul {
        false_description,
        true_description,
        ..
    } = &question.body
    else {
        panic!("a noul body");
    };
    assert_eq!(
        *true_description,
        Some(OrderedJsonValue::Text("it holds".to_owned()))
    );
    assert_eq!(*false_description, None);
}

#[test]
fn should_trim_noul_labels() {
    let question = single_question(
        r#"{"q": {"type": "noul", "instructions": "Holds?",
            "labels": {"false": " no ", "true": "yes"}}}"#,
    );
    let DecisionQuestionBody::Noul { labels, .. } = &question.body else {
        panic!("a noul body");
    };
    assert_eq!(
        *labels,
        Some(NoulLabels {
            false_label: "no".to_owned(),
            true_label: "yes".to_owned(),
        })
    );
}

#[test]
fn should_ignore_fields_it_does_not_know() {
    let question =
        single_question(r#"{"q": {"type": "noul", "instructions": "Holds?", "weight": 3}}"#);
    assert_eq!(question.name, "q");
}

#[test]
fn should_accept_the_largest_allowed_option_count_and_reject_one_more() {
    let at_the_limit = parse_system_one_questions(&ordered_json(&repeated_labels_json(
        MAXIMUM_OPTIONS_PER_QUESTION,
    )));
    assert!(at_the_limit.is_ok());
    assert_eq!(
        parse_error(&repeated_labels_json(MAXIMUM_OPTIONS_PER_QUESTION + 1)),
        QuestionDefinitionError::TooManyOptions {
            question: "q".to_owned(),
            actual: MAXIMUM_OPTIONS_PER_QUESTION + 1,
            maximum: MAXIMUM_OPTIONS_PER_QUESTION,
        }
    );
}

#[test]
fn should_accept_the_largest_allowed_question_count_and_reject_one_more() {
    let mut question_members = Vec::new();
    for question_index in 0..=MAXIMUM_QUESTIONS_PER_REQUEST {
        question_members.push(format!(r#""q{question_index}": {{"type": "noul"}}"#));
    }
    let at_the_limit = format!(
        "{{{}}}",
        question_members[..MAXIMUM_QUESTIONS_PER_REQUEST].join(",")
    );
    let one_over_the_limit = format!("{{{}}}", question_members.join(","));
    assert!(parse_system_one_questions(&ordered_json(&at_the_limit)).is_ok());
    assert_eq!(
        parse_error(&one_over_the_limit),
        QuestionDefinitionError::TooManyQuestions {
            actual: MAXIMUM_QUESTIONS_PER_REQUEST + 1,
            maximum: MAXIMUM_QUESTIONS_PER_REQUEST,
        }
    );
}

#[test]
fn should_reject_questions_that_are_not_a_non_empty_object() {
    assert_eq!(
        parse_error("[]"),
        QuestionDefinitionError::QuestionsNotAnObject { actual: "array" }
    );
    assert_eq!(parse_error("{}"), QuestionDefinitionError::NoQuestions);
}

#[test]
fn should_reject_a_blank_name_and_a_definition_that_is_not_an_object() {
    assert_eq!(
        parse_error(r#"{"  ": {"type": "noul"}}"#),
        QuestionDefinitionError::BlankQuestionName
    );
    assert_eq!(
        parse_error(r#"{"q": "noul"}"#),
        QuestionDefinitionError::DefinitionNotAnObject {
            question: "q".to_owned(),
            actual: "string",
        }
    );
}

#[test]
fn should_reject_a_missing_unknown_or_non_text_type() {
    assert!(matches!(
        parse_error(r#"{"q": {}}"#),
        QuestionDefinitionError::UnknownType { .. }
    ));
    assert!(matches!(
        parse_error(r#"{"q": {"type": "ranking"}}"#),
        QuestionDefinitionError::UnknownType { .. }
    ));
    assert!(matches!(
        parse_error(r#"{"q": {"type": 3}}"#),
        QuestionDefinitionError::UnknownType { .. }
    ));
}

#[test]
fn should_reject_blank_empty_or_boolean_instructions() {
    for invalid_instructions in [r#""   ""#, "[]", "{}", "true"] {
        let questions_json_text =
            format!(r#"{{"q": {{"type": "noul", "instructions": {invalid_instructions}}}}}"#);
        assert!(
            matches!(
                parse_error(&questions_json_text),
                QuestionDefinitionError::InvalidInstructions { .. }
            ),
            "instructions {invalid_instructions} must be rejected"
        );
    }
}

#[test]
fn should_reject_choice_criteria_that_are_missing_empty_or_not_labels() {
    for invalid_criteria in [r#""x""#, "{}", "[]", "[1, 2]"] {
        let questions_json_text =
            format!(r#"{{"q": {{"type": "choice", "criteria": {invalid_criteria}}}}}"#);
        assert!(
            matches!(
                parse_error(&questions_json_text),
                QuestionDefinitionError::InvalidCriteria { .. }
            ),
            "criteria {invalid_criteria} must be rejected"
        );
    }
    assert!(matches!(
        parse_error(r#"{"q": {"type": "choice"}}"#),
        QuestionDefinitionError::InvalidCriteria { .. }
    ));
}

#[test]
fn should_reject_a_repeated_choice_label() {
    assert_eq!(
        parse_error(r#"{"q": {"type": "choice", "criteria": ["a", "b", "a"]}}"#),
        QuestionDefinitionError::RepeatedChoiceLabel {
            question: "q".to_owned(),
            label: "a".to_owned(),
        }
    );
}

#[test]
fn should_reject_score_criteria_that_are_not_a_list_of_described_levels() {
    for invalid_criteria in [r#"{"low": "x"}"#, "[]", r#"["low", null]"#] {
        let questions_json_text =
            format!(r#"{{"q": {{"type": "score", "criteria": {invalid_criteria}}}}}"#);
        assert!(
            matches!(
                parse_error(&questions_json_text),
                QuestionDefinitionError::InvalidCriteria { .. }
            ),
            "criteria {invalid_criteria} must be rejected"
        );
    }
}

#[test]
fn should_reject_noul_criteria_with_other_keys_or_another_shape() {
    for invalid_criteria in [r#"{"yes": "x"}"#, r#"["x"]"#, r#""x""#] {
        let questions_json_text =
            format!(r#"{{"q": {{"type": "noul", "criteria": {invalid_criteria}}}}}"#);
        assert!(
            matches!(
                parse_error(&questions_json_text),
                QuestionDefinitionError::InvalidCriteria { .. }
            ),
            "criteria {invalid_criteria} must be rejected"
        );
    }
}

#[test]
fn should_reject_noul_labels_that_are_not_two_distinct_non_empty_texts() {
    for invalid_labels in [
        r#""x""#,
        r#"{"false": "no"}"#,
        r#"{"false": "same", "true": "same"}"#,
        r#"{"false": " ", "true": "yes"}"#,
        r#"{"false": 0, "true": 1}"#,
        r#"{"false": "no", "true": "yes", "maybe": "perhaps"}"#,
    ] {
        let questions_json_text =
            format!(r#"{{"q": {{"type": "noul", "labels": {invalid_labels}}}}}"#);
        assert!(
            matches!(
                parse_error(&questions_json_text),
                QuestionDefinitionError::InvalidLabels { .. }
            ),
            "labels {invalid_labels} must be rejected"
        );
    }
}

#[test]
fn should_reject_labels_on_a_question_that_is_not_a_noul() {
    assert!(matches!(
        parse_error(
            r#"{"q": {"type": "score", "criteria": ["a", "b"], "labels": {"false": "n", "true": "y"}}}"#
        ),
        QuestionDefinitionError::InvalidLabels { .. }
    ));
}

#[test]
fn should_reject_an_option_order_instead_of_ignoring_it() {
    assert_eq!(
        parse_error(r#"{"q": {"type": "choice", "criteria": ["a", "b"], "option_order": [1, 0]}}"#),
        QuestionDefinitionError::UnsupportedField {
            question: "q".to_owned(),
            field: "option_order",
        }
    );
}

#[test]
fn should_accept_numeric_instructions() {
    let question = single_question(r#"{"q": {"type": "noul", "instructions": 42}}"#);
    assert_eq!(
        question.instructions,
        Some(OrderedJsonValue::SignedInteger(42))
    );
}

#[test]
fn should_keep_the_last_description_of_a_repeated_choice_label() {
    let question = single_question(
        r#"{"q": {"type": "choice", "criteria": {"a": "first", "b": "other", "a": "last"}}}"#,
    );
    let DecisionQuestionBody::Choice { options } = &question.body else {
        panic!("a choice body");
    };
    assert_eq!(options.len(), 2);
    assert_eq!(options[0].key, "a");
    assert_eq!(
        options[0].description,
        Some(OrderedJsonValue::Text("last".to_owned()))
    );
}
