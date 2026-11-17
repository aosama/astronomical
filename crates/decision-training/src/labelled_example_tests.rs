//! Unit tests for the labelled examples format.

use super::*;
use crate::test_support::ROMEO_AND_JULIET_LINES;

fn example_line(labels_json: &str) -> String {
    format!(
        r#"{{"state": "{}",
            "questions": {{
                "speaker": {{"type": "choice", "instructions": "Who speaks?", "criteria": ["Romeo", "Juliet", "Mercutio"]}},
                "longing": {{"type": "score", "instructions": "How much longing?", "criteria": ["none", "some", "great"]}},
                "romantic": {{"type": "noul", "instructions": "Is this line romantic?"}}
            }},
            "labels": {labels_json}}}"#,
        ROMEO_AND_JULIET_LINES[2]
    )
    .replace('\n', " ")
}

#[test]
fn should_read_every_kind_of_label_as_its_canonical_option_index() {
    let line_text = example_line(r#"{"speaker": "Juliet", "longing": 2, "romantic": true}"#);
    let labelled_example = parse_labelled_example(&line_text, 7).unwrap();
    assert_eq!(labelled_example.line_number, 7);
    let correct_indices: Vec<usize> = labelled_example
        .labelled_questions
        .iter()
        .map(|labelled_question| labelled_question.correct_option_index)
        .collect();
    assert_eq!(correct_indices, vec![1, 2, NOUL_TRUE_INDEX]);
    assert_eq!(
        labelled_example.state.as_text(),
        Some(ROMEO_AND_JULIET_LINES[2])
    );
}

#[test]
fn should_read_a_false_noul_label_and_the_lowest_score_level() {
    let line_text = example_line(r#"{"speaker": "Romeo", "longing": 0, "romantic": false}"#);
    let labelled_example = parse_labelled_example(&line_text, 1).unwrap();
    assert_eq!(
        labelled_example.labelled_questions[1].correct_option_index,
        0
    );
    assert_eq!(
        labelled_example.labelled_questions[2].correct_option_index,
        NOUL_FALSE_INDEX
    );
}

#[test]
fn should_skip_blank_lines_and_number_lines_from_one() {
    let first_line = example_line(r#"{"speaker": "Romeo", "longing": 1, "romantic": true}"#);
    let examples_text = format!("\n{first_line}\n   \n{first_line}\n");
    let labelled_examples = parse_labelled_examples(&examples_text).unwrap();
    assert_eq!(labelled_examples.len(), 2);
    assert_eq!(labelled_examples[0].line_number, 2);
    assert_eq!(labelled_examples[1].line_number, 4);
}

#[test]
fn should_refuse_a_file_with_no_examples() {
    assert_eq!(
        parse_labelled_examples("\n  \n"),
        Err(LabelledExampleError::NoExamples)
    );
}

#[test]
fn should_refuse_a_line_that_is_not_json_or_not_an_object() {
    assert!(matches!(
        parse_labelled_example("Romeo, Romeo", 3),
        Err(LabelledExampleError::NotJson { line_number: 3, .. })
    ));
    assert_eq!(
        parse_labelled_example("[1, 2]", 4),
        Err(LabelledExampleError::NotAnObject { line_number: 4 })
    );
}

#[test]
fn should_refuse_a_line_missing_its_labels() {
    let line_text = format!(
        r#"{{"state": "{}", "questions": {{"romantic": {{"type": "noul", "instructions": "Romantic?"}}}}}}"#,
        ROMEO_AND_JULIET_LINES[4]
    );
    assert_eq!(
        parse_labelled_example(&line_text, 1),
        Err(LabelledExampleError::MissingMember {
            line_number: 1,
            member_name: "labels"
        })
    );
}

#[test]
fn should_refuse_a_question_without_a_label() {
    let line_text = example_line(r#"{"speaker": "Romeo", "longing": 1}"#);
    assert_eq!(
        parse_labelled_example(&line_text, 1),
        Err(LabelledExampleError::UnlabelledQuestion {
            line_number: 1,
            question: "romantic".to_owned()
        })
    );
}

#[test]
fn should_refuse_a_label_for_a_question_that_does_not_exist() {
    let line_text =
        example_line(r#"{"speaker": "Romeo", "longing": 1, "romantic": true, "romantik": true}"#);
    assert_eq!(
        parse_labelled_example(&line_text, 1),
        Err(LabelledExampleError::LabelForUnknownQuestion {
            line_number: 1,
            question: "romantik".to_owned()
        })
    );
}

#[test]
fn should_refuse_labels_that_name_no_allowed_answer() {
    for unusable_labels in [
        r#"{"speaker": "Tybalt", "longing": 1, "romantic": true}"#,
        r#"{"speaker": 1, "longing": 1, "romantic": true}"#,
        r#"{"speaker": "Romeo", "longing": 3, "romantic": true}"#,
        r#"{"speaker": "Romeo", "longing": -1, "romantic": true}"#,
        r#"{"speaker": "Romeo", "longing": 1.5, "romantic": true}"#,
        r#"{"speaker": "Romeo", "longing": 1, "romantic": "yes"}"#,
    ] {
        let parse_outcome = parse_labelled_example(&example_line(unusable_labels), 1);
        assert!(
            matches!(
                parse_outcome,
                Err(LabelledExampleError::UnusableLabel { .. })
            ),
            "{unusable_labels} was accepted"
        );
    }
}

#[test]
fn should_report_malformed_questions_with_their_line() {
    let line_text = format!(
        r#"{{"state": "{}", "questions": {{}}, "labels": {{}}}}"#,
        ROMEO_AND_JULIET_LINES[5]
    );
    assert!(matches!(
        parse_labelled_example(&line_text, 9),
        Err(LabelledExampleError::InvalidQuestions { line_number: 9, .. })
    ));
}

#[test]
fn should_count_questions_and_count_a_repeated_text_once() {
    let line_text = example_line(r#"{"speaker": "Romeo", "longing": 1, "romantic": true}"#);
    let labelled_examples =
        parse_labelled_examples(&format!("{line_text}\n{line_text}\n")).unwrap();

    assert_eq!(
        labelled_data_counts(&labelled_examples),
        LabelledDataCounts {
            labelled_question_count: 6,
            distinct_state_count: 1,
        }
    );
}

#[test]
fn should_count_nothing_in_no_examples() {
    assert_eq!(
        labelled_data_counts(&[]),
        LabelledDataCounts {
            labelled_question_count: 0,
            distinct_state_count: 0,
        }
    );
}
